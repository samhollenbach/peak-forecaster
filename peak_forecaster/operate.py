import datetime
import numpy as np
from tariff.bill_calculator import BillCalculator
import pandas as pd
from peak_forecaster import process

class StandardOperator:

    def __init__(self, site, start, end):
        """

        :param str site:
        :param Date start:
        :param Date end:
        """
        self.site = site
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        # TODO: Change from site to site
        self.lt_capacity = 155.72
        self.tank_min = 0
        self.bill_calculator = BillCalculator('pge')

    def get_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_total_bill(baseline_load, load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_total_bill(target_load, load_column='target')
        return base_bill - target_bill

    def get_demand_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_demand_bill(baseline_load, load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_demand_bill(target_load, load_column='target')
        return base_bill - target_bill

    def get_energy_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_energy_bill(baseline_load, load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_energy_bill(target_load, load_column='target')
        return base_bill - target_bill

    def prepare_data(self, data):
        """
        Add offsets, soc, and target field placeholders

        :param data:
        :return:
        """
        new_data = []
        if not isinstance(data, list):
            data = process.group_data(data)
        for day in data:
            day['offsets'] = 0
            day['soc'] = 0
            day['target'] = day['building_baseline']
            new_data.append(day)
        return new_data


    def run_standard_operation(self, time_start, time_end, data, predictions=None, crs_predictions=None, thresholds=None, threshold_matching=True):

        if thresholds is None and (predictions is None or crs_predictions is None):
            raise ValueError("Standard operational strategy needs either peak predictions or target thresholds provided")
        elif thresholds is None:
            # TODO: Fake thresholds - replace with actual equation
            thresholds = [p[0] - crs[0] for p, crs in zip(predictions, crs_predictions)]

        highest_threshold = thresholds[0]

        # Add necessary columns and group data by days if needed
        daily_data = self.prepare_data(data)

        if len(daily_data) != len(thresholds):
            raise ValueError("Number of days in data and amount of thresholds do not match")


        new_day_data = []
        for i, day in enumerate(daily_data):
            current_soc = 0

            # Follow the highest threshold set
            # even if we haven't reached a target that high
            if threshold_matching:
                if thresholds[i] > highest_threshold:
                    highest_threshold = thresholds[i]

            # Add predictions to data for display
            pred = predictions[i][0]
            if predictions is not None:
                day['peak_prediction'] = pred
            pred_crs = crs_predictions[i][0]
            if crs_predictions is not None:
                day['crs_prediction'] = pred_crs

            # What the predicted threshold was for display
            day['threshold_old'] = thresholds[i]

            # What the actual followed threshold is for action
            day['threshold'] = highest_threshold

            # Shrink activity window if predicted threshold
            # is significantly below the current threshold
            time_start_corrected = time_start
            threshold_buffer = pred - (pred_crs/2)
            if threshold_buffer < highest_threshold:
                dif = (highest_threshold - threshold_buffer) / highest_threshold
                h = int(time_start.hour)
                m = int(time_start.minute)
                t = h * 60 + m
                t += np.floor(2000 * dif)
                t = min(t, 20 * 60)
                h = np.floor(t / 60)
                m = np.floor(t - (h * 60))
                time_start_corrected = datetime.time(int(h), int(m))

            print(time_start_corrected)

            # Run operation
            for index, row in day.iterrows():
                ts = row['timestamp']

                day.at[index, 'threshold'] = highest_threshold
                # If not in testing window
                if ts.time() < time_start_corrected or ts.time() > time_end:
                    continue

                current_load = row['building_baseline']

                # Get new SOC and offset for this interval
                soc, offset = self.get_next_soc_offset(
                    current_load,
                    current_soc,
                    row['charge_limits'],
                    row['discharge_limits'],
                    row['cop_charge'],
                    row['cop_discharge'],
                    row['heat_leak'],
                    row['threshold'])
                day.at[index, 'soc'] = soc
                day.at[index, 'offsets'] = offset
                target = current_load - offset
                if target > highest_threshold:
                    highest_threshold = target
                day.at[index, 'target'] = target

                current_soc = soc

            new_day_data.append(day)
        new_data = pd.concat(new_day_data)
        return new_data


    def get_next_soc_offset(self, current_load, current_soc, chg_limit, dchg_limit,
                            chg_cop, dchg_cop, heat_leak, threshold):
        threshold_offset = current_load - threshold

        # FOR HEAT LEAK BEFORE
        current_soc -= heat_leak

        # TODO: Check if COPs need the factor of 4

        if threshold_offset == 0:  # DMT mode
            offset = 0
            soc = current_soc
        elif threshold_offset < 0:  # CHG mode
            offset = max(threshold_offset, -chg_limit)
            soc = current_soc - (offset * chg_cop) / 4
            if soc > self.lt_capacity:
                offset = (self.lt_capacity - current_soc) / chg_cop * 4
                soc = self.lt_capacity
        else:  # DCHG mode
            offset = min(threshold_offset, dchg_limit)
            soc = current_soc - (offset * dchg_cop) / 4
            if soc < self.tank_min:
                offset = (self.tank_min - current_soc) / dchg_cop * 4
                soc = self.tank_min
        return soc, offset


    # def run_timeclock(self, data, initial_soc):
    #     soc_profile = [initial_soc]
    #     offset_profile = []
    #     state_profile = []
    #
    #     for index, row in data.iterrows():
    #         period = row['periods']
    #         current_soc = soc_profile[-1]
    #         if "On-Peak" in period:
    #             offset = row['dchg_limit']
    #             cop = row['discharge_cop']
    #             if current_soc > self.tank_min:
    #                 next_soc = current_soc - offset * cop / 4
    #                 if next_soc < self.tank_min:
    #                     corrected_offset = (current_soc) * 4 / cop
    #                     offset_profile.append(corrected_offset)
    #                     state_profile.append("CHG")
    #                     soc_profile.append(self.tank_min)
    #                 else:
    #                     offset_profile.append(offset)
    #                     state_profile.append("DCHG")
    #                     soc_profile.append(current_soc - offset * cop / 4)
    #             else:
    #                 offset_profile.append(0)
    #                 state_profile.append("DMT")
    #                 soc_profile.append(current_soc)
    #
    #         else:
    #             offset = -row['chg_limit']
    #             cop = row['charge_cop']
    #             if current_soc < self.rb_capacity:
    #                 next_soc = current_soc - offset * cop / 4
    #                 if next_soc > self.rb_capacity:
    #                     corrected_offset = -(self.rb_capacity - current_soc) * 4 / cop
    #                     offset_profile.append(corrected_offset)
    #                     state_profile.append("CHG")
    #                     soc_profile.append(self.rb_capacity)
    #                 else:
    #                     offset_profile.append(offset)
    #                     state_profile.append("CHG")
    #                     soc_profile.append(next_soc)
    #             else:
    #                 offset_profile.append(0)
    #                 state_profile.append("DMT")
    #                 soc_profile.append(current_soc)
    #
    #     return soc_profile, offset_profile, state_profile

