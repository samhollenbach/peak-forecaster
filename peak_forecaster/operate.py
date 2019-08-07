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

        cap_info = pd.read_csv('../input/WM_LTSB_mass_and_SST_new.csv')
        cap = cap_info.loc[cap_info['Store'] == site]['mass_total']
        self.lt_capacity = float(cap)
        self.tank_min = 0
        self.bill_calculator = BillCalculator('pge')

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


    def run_standard_operation(self, time_start, time_end, data,
                               predictions=None, crs_predictions=None,
                               thresholds=None, threshold_matching=True,
                               dynamic_precool=True, threshold_derating=0.7):
        """

        :param Time time_start:
        :param Time time_end:
        :param Dataframe data:
        :param list predictions:
        :param list crs_predictions:
        :param list thresholds:
        :param bool threshold_matching:
        :param bool dynamic_precool:
        :return:
        """


        if thresholds is None and (predictions is None or crs_predictions is None):
            raise ValueError("Standard operational strategy needs either peak predictions or target thresholds provided")
        elif thresholds is None:
            thresholds = [p[0] - (crs[0] * threshold_derating) for p, crs in zip(predictions, crs_predictions)]

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
            if dynamic_precool:
                threshold_buffer = pred - (pred_crs/2)
                if threshold_buffer < highest_threshold:
                    dif = (highest_threshold - threshold_buffer) / highest_threshold
                    h = time_start.hour
                    m = time_start.minute
                    t = h * 60 + m
                    t += np.floor(4000 * dif)
                    t = min(t, 20 * 60)
                    h = np.floor(t / 60)
                    m = np.floor(t - (h * 60))
                    time_start_corrected = datetime.time(int(h), int(m))

            ##############################
            ### Run Standard Operation ###
            ##############################
            for index, row in day.iterrows():
                ts = row['timestamp']



                day.at[index, 'threshold'] = highest_threshold

                # If not in testing window
                if ts.time() < time_start_corrected or ts.time() > time_end:
                    if row['target'] > highest_threshold:
                        highest_threshold = row['target']
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
                    highest_threshold)
                day.at[index, 'soc'] = soc
                day.at[index, 'offsets'] = offset
                target = current_load - offset

                day.at[index, 'target'] = target
                if target > highest_threshold:
                    highest_threshold = target

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
                offset = (current_soc - self.lt_capacity) / chg_cop * 4
                soc = self.lt_capacity
        else:  # DCHG mode
            offset = min(threshold_offset, dchg_limit)
            soc = current_soc - (offset * dchg_cop) / 4
            if soc < self.tank_min:
                offset = (current_soc - self.tank_min) / dchg_cop * 4
                soc = self.tank_min
        return soc, offset
