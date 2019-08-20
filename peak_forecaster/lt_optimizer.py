"""
This module encompasses the core functionality of the schedule optimizer. The
Optimizer class holds the engine to Linear programming and generating a target
building load schedule to be sent to the reactive planner service.

"""
# System imports
import copy
from collections import OrderedDict

# 3rd party imports
import pulp
import pandas
from tariff import Tariff, NON_COINCIDENT

# Project imports
from optimizer_engine import cop, time_ops, common
from optimizer_engine.operating_limits import (get_discharge_limits,
                                               get_charge_limits)


class Optimizer:
    """Create & solve linear optimization to genereate target building load

    The Optimizer class constructs a linear optimization with multiple input
    datapoints and configurations. It has methods to create & solve an LP
    optimization & extract decision variables to ultimately return a target
    building load.

    Attributes:
        start (datetime): first timestamp of the optimization window
        end (datetime): last timestamp of the optimization window
        mcl (int): Max charge limit (fixed value)
        mdl (int): Max discharge limit (fixed value)
        rb_capacity (int): RB tank capacity in KWH_thermal
        soc_init (int): Initial state of charge of the RB tank
        big_m (int): Big M value for slack variables
        time_labels (list): time labels [T01, T02..]
        timestamp_labels (dict): timestamps associated with each time label
        soc (dict): Decision variable - state of charge of the RB, float
        dof (dict): Decision variable - discharge offsets, float
        cof (dict): Decision variable - charge offsets, float
        y (dict): Decision variable - selector variable, binary
        max_peaks (dict): dummy variables, peak demand for all periods
        frame (obj): PuLP LP optimization object
    """

    def __init__(self, config, optimize_energy=True):
        """
        Initialize the Optimizer object.

        :param dict config: dictionary with input configurations
        """
        self.tariff_id = config.get('tariff_id') or config['site_id']
        self.tariff = Tariff(self.tariff_id)

        self.start = config['start']
        self.end = config['end']

        self.optimize_energy = optimize_energy

        # Load performance parameters from configuration
        self.mcl = config['MCL']
        self.mdl = config['MDL']
        self.mrc = config['MRC']
        self.rb_capacity = config['RB_capacity']
        self.soc_init = config['SOC_initial']
        self.soc_final = config.get('SOC_final', self.soc_init)
        self.rb_min_soc = config.get('RB_min_soc', 0)
        self.min_charge_offset = config['min_charge_offset']
        self.min_discharge_offset = config['min_discharge_offset']
        self.rte_setpoint = config.get('RTE_setpoint')
        self.cop_dchg_coefficients = config['cop_dchg_coefficients']
        self.cop_chg_coefficients = config['cop_chg_coefficients']
        self.heat_leak_coefficients = config.get('heat_leak_coefficients')
        self.chg_limit_curve = config.get('chg_limit_curve')
        self.dchg_limit_curve = config.get('dchg_limit_curve')
        self.demand_peaks = config.get('peaks')

        # Create peak structure with zero values of all peaks if peaks are
        # not included
        if not self.demand_peaks:
            self.demand_peaks = copy.deepcopy(self.tariff.demand_rates())
            for season in self.demand_peaks:
                for peak in self.demand_peaks[season]:
                    self.demand_peaks[season][peak] = 0

        # Load selectable constraints and output
        self.constraints = config.get('constraints')
        self.outputs = config.get('outputs')

        # Load big M variable value
        self.big_m = config['M']

        # Fetch dataframe wih 15 min interval timestamps and time labels
        self.timestamp_df = time_ops.get_timestamps(self.start, self.end)

        # Create list of time labels
        self.time_labels = list(self.timestamp_df.time_label)

        # Create mapping of time labels to timestamps
        self.timestamp_labels = dict(
            zip(self.timestamp_df.time_label, self.timestamp_df.timestamp))

        # Define a frame & set it up as an LP Minimization problem
        self.frame = pulp.LpProblem(
            "Simple schedule optimization",
            pulp.LpMinimize)

        # Decision variables
        self.soc = None
        self.dof = None
        self.cof = None
        self.y = None
        self.max_peaks = None

        # Define a dictionary to hold the final values for the decision
        # variables once the optimizer has solved the problem
        self.decision_variables = {}

        # Constraints
        self.chg_limits = None
        self.dchg_limits = None
        self.cop_dchg = None
        self.cop_chg = None

    def _define_decision_variables(self):
        """
        Define attributes related to optimization framework.

        This function utilizes the attributes set during initialization to set
        more attributes, namely the decision variables & other slack or dummy
        variables. The prime deicison variables include offset variables, SOC
        and a selector variable. The dummy variables include variables for each
        of the demand charges. Finally this function also sets up the frame for
        the optimization problem and defines it specifically as a minimization
        LP problem.
        """
        # Define SOC decision variables (continuous)
        self.soc = pulp.LpVariable.dicts(
            "SOC",
            self.time_labels,
            lowBound=self.rb_min_soc,
            upBound=self.rb_capacity)

        # Define discharge offset decision variables (continuous)
        self.dof = pulp.LpVariable.dicts(
            "Discharge_Offset",
            self.time_labels,
            lowBound=0,
            upBound=self.mdl)

        # Define charge offset decision variables (continuous)
        self.cof = pulp.LpVariable.dicts(
            "Charge_Offset",
            self.time_labels,
            lowBound=0,
            upBound=None)

        # Define selector decision variables (binary)
        self.y = pulp.LpVariable.dicts(
            "Selector",
            self.time_labels,
            lowBound=None,
            upBound=None,
            cat='Binary')

        # Define variables max peak, part-peak & non-coincidental demand Summer
        self.max_peaks = {season: {peak: pulp.LpVariable(
            "Maximum {} {}".format(peak, season), lowBound=0, upBound=None)
            for peak in self.demand_peaks[season]}
            for season in self.demand_peaks}

    def _define_objective(self, df, building_power_column='building_baseline'):
        """
        Define the objective function of the LP Minimization

        This function takes in building power as a dataframe, loads the energy
        tariff and demand charges from the tariffs module and proceeds to setup
        the objective function in the frame attribute of the optimizer object.
        The objective function here is simply the total bill including the
        energy and demand charges as a function of the building power, offset
        values and the peak demands.

        :param DataFrame df: Building power time series inputs
        :param building_power_column: Name of building baseline column
        """
        # Fetch energy tariff's for the given time frame
        energy_tariff_df = self.tariff.apply_energy_rates(df)

        # Fetch demand charges
        demand_charges = self.tariff.demand_rates()

        # Create mapping of energy tariffs and time labels for the given frame
        energy_tariff = dict(
            zip(self.time_labels, energy_tariff_df.energy_tariff))

        # Create mapping of building power and time labels for the given frame
        building_power = dict(
            zip(self.time_labels, df[building_power_column]))

        # ===== Set objective function =====
        # Demand portion
        season_df = self.tariff.apply_season(df)
        demand = []
        for season in self.max_peaks:
            # Determine the fraction of rows in the current season
            n = len(season_df[season_df['season'] == season])
            fraction = n / len(season_df)

            # Weight each peak by the demand charge in that period as well
            # as the fraction of days in the current season
            for period, rate in demand_charges[season].items():
                peak = self.max_peaks[season][period]
                demand.append(peak * rate * fraction)

        # Energy portion
        energy = []
        for t in self.time_labels:
            offset = self.dof[t] - self.cof[t]
            power = building_power[t] - offset
            energy.append(energy_tariff[t] * power / 4)

        # Add objective to frame
        if self.optimize_energy:
            print("Running Optimizer with Demand / Energy optimization")
            self.frame += pulp.lpSum(demand + energy)
        else:
            print("Running Optimizer with only Demand optimization")
            self.frame += pulp.lpSum(demand)

    def _add_constraints(self, df, building_power_column='building_baseline',
                         crs_power_column='crs_baseline',
                         temperature_column='temperature',
                         discharge_limit_column='discharge_limits',
                         charge_limit_column='charge_limits',
                         cop_charge_column='cop_charge',
                         cop_discharge_column='cop_discharge'):
        """
        Define constraints for the LP Minimization

        This function takes in building power as a DataFrame and adds energy
        balance constraints to the optimization frame, as well as limits on
        different parameters including both offsets to limit their value as
        well as force only one parameter to be active in each 15 min interval.

        :param DataFrame df: Building power, CRS power, and OAT time series
        :param str building_power_column: Name of building baseline column
        :param str crs_power_column: Name of CRS baseline column
        :param str temperature_column: Name of air temperature column
        :param str discharge_limit_column: Name of discharge limit column
        :param str charge_limit_column: Name of charge limit column
        """
        # If the CRS power is given, use it to derive CHG/DCHG limits,
        # otherwise the charge and discharge limits must be given
        if crs_power_column in df.columns:
            # CRS power time series inputs
            crs_power_df = df[crs_power_column]

            # Derive discharge limits from CRS power
            self.dchg_limits = get_discharge_limits(crs_power_df)

            # Derive charge limits from CRS power
            self.chg_limits = get_charge_limits(crs_power_df, self.mrc)
        else:
            # Discharge limit time series inputs
            self.dchg_limits = df[discharge_limit_column]

            # Charge limit power time series inputs
            self.chg_limits = df[charge_limit_column]

        # Fetch max compressor capacity for each interval based on temperature
        # mcc_df = cop.generate_mcc(temperature_df)

        # Create mappings of building power and time labels
        building_power = dict(zip(self.time_labels, df[building_power_column]))

        # Create mappings of time series discharge limits and time labels
        dchg_limits = dict(zip(self.time_labels, self.dchg_limits))

        # Create mappings of time series charge limits and time labels
        chg_limits = dict(zip(self.time_labels, self.chg_limits))

        # Create COP time series
        if (cop_charge_column in df.columns and
                cop_discharge_column in df.columns):
            self.cop_chg = df[cop_charge_column]
            self.cop_dchg = df[cop_discharge_column]
        elif cop_charge_column in df.columns:
            self.cop_chg = df[cop_charge_column]

            # Generate CHG COP for each interval based on temperature
            cop_df = cop.generate_discharge_cop(
                df, self.cop_dchg_coefficients,
                temperature_column=temperature_column)
            self.cop_dchg = cop_df['cop_dchg']
        elif cop_discharge_column in df.columns:
            self.cop_dchg = df[cop_discharge_column]

            # Generate CHG COP for each interval based on temperature
            cop_df = cop.generate_charge_cop(
                df, self.cop_chg_coefficients,
                temperature_column=temperature_column)
            self.cop_chg = cop_df['cop_chg']
        else:
            # Generate CHG & DCHG COP for each interval based on temperature
            cop_df = cop.generate_cop(
                df, self.cop_dchg_coefficients, self.cop_chg_coefficients,
                temperature_column=temperature_column)
            self.cop_dchg = cop_df['cop_dchg']
            self.cop_chg = cop_df['cop_chg']

        # Create mapping of discharge COP's and time labels
        cop_dchg = dict(zip(self.time_labels, self.cop_dchg))

        # Create mapping of charge COP's and time labels
        cop_chg = dict(zip(self.time_labels, self.cop_chg))

        if self.heat_leak_coefficients:
            heat_leak_df = cop.generate_heat_leak(
                df, self.heat_leak_coefficients,
                temperature_column=temperature_column)
        else:
            heat_leak_df = df.copy()
            # heat_leak_df['heat_leak'] = 0

        self.heat_leak = heat_leak_df['heat_leak']
        # Create mapping of charge COP's and time labels
        heat_leak = dict(zip(self.time_labels, self.heat_leak))

        # Run loop to add constraints to the frame
        common.timer('constraints_loop')
        for t in range(len(self.time_labels)):
            if t < len(self.time_labels) - 1:
                # Difference in subsequent SOC states in kWh
                soc_difference = (self.soc[self.time_labels[t]] -
                                  self.soc[self.time_labels[t + 1]])

                # Energy fed into or used up from the tank
                offset_energy = 0.25 * (cop_dchg[self.time_labels[t]] *
                                        self.dof[self.time_labels[t]] -
                                        cop_chg[self.time_labels[t]] *
                                        self.cof[self.time_labels[t]])

                energy_heat_leak = 0.25 * (self.soc[self.time_labels[t]] *
                                           heat_leak[self.time_labels[t]])

                # Energy balance equation for the RB
                effective_energy = offset_energy + energy_heat_leak
                self.frame += soc_difference == effective_energy

                # ===== OPTIONAL CONSTRAINT =====
                if self.constraints['time_transition']:
                    # Helper variable
                    beta = (self.y[self.time_labels[t]] -
                            self.y[self.time_labels[t + 1]])

                    # Equations to support time transitions
                    self.frame += (self.dof[self.time_labels[t + 1]] <=
                                   self.big_m * (beta + 1))

                    self.frame += (self.cof[self.time_labels[t + 1]] <=
                                   self.big_m * (1 - beta))

                # Helper variable
                alpha = (self.y[self.time_labels[t]] +
                         self.y[self.time_labels[t + 1]])

                # ===== OPTIONAL CONSTRAINT =====
                if self.constraints['minimum_discharge_offset']:
                    # Equation to support minimum discharge offset
                    self.frame += (self.dof[self.time_labels[t + 1]] >=
                                   self.min_discharge_offset * (alpha - 1))

                # ===== OPTIONAL CONSTRAINT =====
                if self.constraints['minimum_charge_offset']:
                    # Equation to support minimum charge offset
                    self.frame += (self.cof[self.time_labels[t + 1]] >=
                                   self.min_charge_offset * (1 - alpha))

            # Constraint discharge offset less than time series dchg limits
            self.frame += (self.dof[self.time_labels[t]] <=
                           dchg_limits[self.time_labels[t]])

            # Constrain charge offset less than time series CRS charge limits
            self.frame += (self.cof[self.time_labels[t]] <=
                           chg_limits[self.time_labels[t]])

            # Constraints that allow either chg or dchg offset in an interval
            self.frame += (self.dof[self.time_labels[t]] <=
                           self.big_m * self.y[self.time_labels[t]])

            self.frame += (self.cof[self.time_labels[t]] <=
                           self.big_m * (1 - self.y[self.time_labels[t]]))

            # Once the RB has chosen to charge or discharge, effective
            # demand represents the adjustment to the building baseline by
            # either a charge or discharge offset to create a new target for
            #  the given 15-minute interval
            effective_demand = (building_power[self.time_labels[t]] -
                                self.dof[self.time_labels[t]] +
                                self.cof[self.time_labels[t]])

            # Since the operation of finding a maximum is not linear,
            # constraints must be added to select the maximum effective demand
            # amongst all 15-minute intervals
            time = self.timestamp_labels[self.time_labels[t]]
            season = self.tariff.season(time)
            period = self.tariff.period(time)

            if NON_COINCIDENT in self.max_peaks[season]:
                peak = self.max_peaks[season][NON_COINCIDENT]
                self.frame += peak >= effective_demand

            if period in self.max_peaks[season]:
                peak = self.max_peaks[season][period]
                self.frame += peak >= effective_demand

            # ===== OPTIONAL CONSTRAINT =====
            if self.constraints['dchg_limit_curve'] and self.dchg_limit_curve:
                if not self.dchg_limit_curve:
                    raise TypeError('Value for "dchg_limit_curve" unset')
                # Additional bounding constraint on discharge offset as f(SOC)
                for line in self.dchg_limit_curve:
                    m, b = line
                    self.frame += self.dof[self.time_labels[t]] <= ((m * self.soc[self.time_labels[t]] + b) / cop_dchg[self.time_labels[t]])
            else:
                if self.dchg_limit_curve:
                    common.warning('Value provided for "dchg_limit_curve" but '
                                   '"dchg_limit_curve" constraint is not '
                                   'active')

            # ===== OPTIONAL CONSTRAINT =====
            if self.constraints['chg_limit_curve'] and self.chg_limit_curve:
                if not self.chg_limit_curve:
                    raise TypeError('Value for "chg_limit_curve" unset')
                # Additional bounding constraint on discharge offset as f(SOC)
                for line in self.chg_limit_curve:
                    m, b = line
                    self.frame += self.cof[self.time_labels[t]] <= ((m * self.soc[self.time_labels[t]] + b) / cop_chg[self.time_labels[t]])
            else:
                if self.chg_limit_curve:
                    common.warning('Value provided for "chg_limit_curve" but '
                                   '"chg_limit_curve" constraint is not '
                                   'active')

        common.timer('constraints_loop', "Ran constraints loop")

        # Constraint to set initial SOC in the frame
        self.frame += self.soc["T01"] == self.soc_init

        # Constraint to set final SOC in the frame (same as initial)
        self.frame += self.soc[self.time_labels[-1]] == self.soc_final

        # ===== OPTIONAL CONSTRAINT =====
        if self.constraints['fixed_rte']:
            if not self.rte_setpoint:
                raise TypeError('Value for "rte_setpoint" unset')
            # Constraint to force RTE to a specific value
            total_dof = sum([self.dof[t] for t in self.time_labels])
            total_cof = sum([self.cof[t] for t in self.time_labels])
            self.frame += total_dof >= self.rte_setpoint * total_cof
        else:
            if self.rte_setpoint:
                common.warning('Value provided for "rte_setpoint" but '
                               '"fixed_rte" constraint is not active')

        # Constraint to set target peaks based on historical peaks in the
        # billing period
        for season in self.max_peaks:
            for period in self.max_peaks[season]:
                peak = self.max_peaks[season][period]
                self.frame += peak >= self.demand_peaks[season][period]

                # Ensure NON_COINCIDENT peak is tied to all other peaks
                if period == NON_COINCIDENT:
                    for other_period in self.max_peaks[season]:
                        if other_period != NON_COINCIDENT:
                            other_peak = self.max_peaks[season][other_period]
                            self.frame += peak >= other_peak

    def write_frame(self, file_name):
        """
        Write linear programming framework to a file

        This function writes the LP framework developed by form_optimization
        to the file path specified in the argument.

        :param str file_name:: file path for storing lp framework
        """
        # Write frame LP problem to file
        self.frame.writeLP(file_name)

    def _solve_frame(self):
        """
        Solve optimization frame and time it

        This function solves the frame setup by the form_optimization method &
        prints out the status of the solution (solved, undefined, infeasible).
        It also prints out the time taken for convergence.
        """
        pulp.LpSolverDefault.msg = 1

        # Solve & time frame
        common.timer('solve_frame')

        self.frame.solve(pulp.PULP_CBC_CMD(maxSeconds=120))
        common.timer('solve_frame', "Convergence time")

        # Post solution status to logger
        status = pulp.LpStatus[self.frame.status]
        common.debug("Solution status: {}".format(status))

        # Extract decision variable solutions from solved frame
        self.decision_variables = {v.name: v.varValue
                                   for v in self.frame.variables()}

    def _get_target(self, df, building_power_column='building_baseline',
                    temperature_column='temperature',
                    time_column='timestamp'):
        """
        Construct the target building load from the offsets & baseline

        This function takes in building power and generates the target building
        load by adding the offset values to the building baseline power.

        :param DataFrame df: Building power, CRS power, and OAT time series
        :param str building_power_column: Name of building baseline column
        :param str temperature_column: Name of air temperature column
        :param time_column: Name of time column
        :return DataFrame: Target building time series output with offsets
        """
        # Calculate effective offsets (Discharge - Charge)
        offset_values = []
        soc_values = []
        for label in self.time_labels:
            dof = self.decision_variables["Discharge_Offset_" + label]
            cof = self.decision_variables["Charge_Offset_" + label]
            soc = self.decision_variables["SOC_" + label]

            offset_values.append(dof - cof)
            soc_values.append(soc)

        # Calculate building target value based on baseline and offset
        building_target = [
            a - b for a, b in zip(df[building_power_column], offset_values)]

        output_map = OrderedDict([
            ('timestamp', df[time_column]),
            ('baseline', df[building_power_column]),
            ('offsets', offset_values),
            ('load_values', building_target),
            ('soc', soc_values),
            ('charge_limits', self.chg_limits),
            ('discharge_limits', self.dchg_limits),
            ('cop_dchg', self.cop_dchg),
            ('cop_chg', self.cop_chg),
            ('heat_leak', self.heat_leak),
            ('temperature', df[temperature_column]),
        ])

        return pandas.DataFrame(OrderedDict(
            [(output, output_map[output]) for output in output_map
             if self.outputs.get(output)]
        ))

    def solve(self, df, building_power_column='building_baseline',
              crs_power_column='crs_baseline',
              temperature_column='temperature',
              discharge_limit_column='discharge_limits',
              charge_limit_column='charge_limits',
              cop_charge_column='cop_charge',
              cop_discharge_column='cop_discharge'):
        """
        Constructs the LP Minimization framework by creating an LPP minimizer
        frame, adding an objective function, and applying constraints to it.
        The solver then produces an optimal target schedule for the RB.

        :param DataFrame df: Building power, CRS power, and OAT time series
        :param str building_power_column: Name of building baseline column
        :param str crs_power_column: Name of CRS baseline column
        :param str temperature_column: Name of air temperature column
        :param str discharge_limit_column: Name of discharge limit column
        :param str charge_limit_column: Name of charge limit column
        :return DataFrame: Target building time series output with offsets
        """
        # Call function to define decision variables
        self._define_decision_variables()

        # Call function to define objective function
        self._define_objective(df, building_power_column=building_power_column)

        # Call function to add constraints to frame
        self._add_constraints(df,
                              building_power_column=building_power_column,
                              crs_power_column=crs_power_column,
                              temperature_column=temperature_column,
                              discharge_limit_column=discharge_limit_column,
                              charge_limit_column=charge_limit_column,
                              cop_charge_column=cop_charge_column,
                              cop_discharge_column=cop_discharge_column)

        # Run the solver
        self._solve_frame()

        return self._get_target(df)
