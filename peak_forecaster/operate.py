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

        if site.startswith('WF'):
            site_file = '../input/WF_LTSB_mass_and_SST.csv'
        else:
            site_file = '../input/WM_LTSB_mass_and_SST_new.csv'
        cap_info = pd.read_csv(site_file)
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
                               dynamic_precool=True, threshold_derating=None,
                               reset_soc=False, start_soc=0.5,
                               start_nth_threshold=None):
        """

        :param Time time_start:
        :param Time time_end:
        :param Dataframe data:
        :param dict predictions:
        :param dict crs_predictions:
        :param dict thresholds:
        :param bool threshold_matching:
        :param bool dynamic_precool:
        :return:
        """



        if thresholds is None and (predictions is None or crs_predictions is None):
            raise ValueError("Standard operational strategy needs either peak predictions or target thresholds provided")
        elif thresholds is None:
            thresholds = {}
            for period in predictions.keys():
                # Use period based threshold derating
                if threshold_derating is None or period not in threshold_derating:
                    tder = 1.0
                else:
                    tder = threshold_derating[period]
                # Calculate thresholds from peak and crs predictions
                thresholds[period] = [p[0] - (crs[0] * tder)
                                      for p, crs in zip(predictions[period], crs_predictions[period])]

        # Store period keys for reference
        period_keys = list(thresholds.keys())

        # Add necessary columns and group data by days if needed
        daily_data = self.prepare_data(data)

        for period in period_keys:
            if len(daily_data) != len(thresholds[period]):
                raise ValueError(f"Number of days in data and amount of"
                                 f" {period} thresholds do not match")


        # Set starting thresholds to first threshold in period
        highest_threshold = {}
        for period, trs in thresholds.items():
            for first_thresh in trs:
                if not np.isnan(first_thresh):
                    highest_threshold[period] = first_thresh
                    continue

        # TODO: Implement per period
        if start_nth_threshold is not None:
            sorted_thresholds = sorted(thresholds, reverse=True)
            highest_threshold = sorted_thresholds[start_nth_threshold-1]


        new_day_data = []
        current_soc = start_soc * self.lt_capacity

        ####################
        # Begin Daily Loop #
        ####################
        for i, day in enumerate(daily_data):

            # Get active threshold keys that are present for this day
            period_keys_day = np.intersect1d(day['period'].unique(), period_keys)

            if reset_soc:
                current_soc = 0

            if predictions is not None:
                pred = predictions['Non-Coincident'][i][0]
                day['peak_prediction'] = pred
            if crs_predictions is not None:
                pred_crs = crs_predictions['Non-Coincident'][i][0]
                day['crs_prediction'] = pred_crs


            for period in period_keys_day:

                # Follow the highest threshold set
                # even if we haven't reached a target that high
                if threshold_matching:
                    if thresholds[period][i] > highest_threshold[period]:
                        highest_threshold[period] = thresholds[period][i]
                    # If another period breaks the Non-Coincident peak, set that new peak
                    if highest_threshold[period] > highest_threshold['Non-Coincident']:
                        highest_threshold['Non-Coincident'] = highest_threshold[period]

                # Add predictions to data for display
                if predictions is not None:
                    pred = predictions[period][i][0]
                    day.loc[day['period'] == period, 'peak_prediction'] = pred
                if crs_predictions is not None:
                    pred_crs = crs_predictions[period][i][0]
                    day.loc[day['period'] == period, 'crs_prediction'] = pred_crs

            for all_period in day['period'].unique():

                if all_period in period_keys and not np.isnan(thresholds[all_period][i]):
                    # What the predicted threshold was for display
                    day.loc[day['period'] == all_period, 'threshold_old'] = thresholds[all_period][i]
                    # What the actual followed threshold is for action
                    day.loc[day['period'] == all_period, 'threshold'] = highest_threshold[all_period]
                else:
                    # What the predicted threshold was for display
                    day.loc[day['period'] == all_period, 'threshold_old'] = thresholds['Non-Coincident'][i]
                    # What the actual followed threshold is for action
                    day.loc[day['period'] == all_period, 'threshold'] = highest_threshold['Non-Coincident']



            # Shrink activity window if predicted threshold
            # is significantly below the current threshold
            time_start_corrected = time_start
            # if dynamic_precool:
            #     threshold_buffer = pred - (pred_crs/2)
            #     # TODO: Implement per period
            #     if threshold_buffer < highest_threshold['Non-Coincident']:
            #         dif = (highest_threshold['Non-Coincident'] - threshold_buffer) / highest_threshold
            #         h = time_start.hour
            #         m = time_start.minute
            #         t = h * 60 + m
            #         t += np.floor(4000 * dif)
            #         t = min(t, 20 * 60)
            #         h = np.floor(t / 60)
            #         m = np.floor(t - (h * 60))
            #         time_start_corrected = datetime.time(int(h), int(m))

            #############################
            ### Iterate 15m Intervals ###
            #############################
            for index, row in day.iterrows():
                ts = row['timestamp']
                period = row['period']
                if period not in period_keys_day:
                    period = 'Non-Coincident'

                day.at[index, 'soc'] = current_soc
                day.at[index, 'threshold'] = highest_threshold[period]

                current_load = row['building_baseline']

                # Apply heat leak even if not operating
                current_soc = self.apply_heat_leak(current_soc, row['heat_leak'])

                # Get new SOC and offset for this interval
                soc, offset = self.get_next_soc_offset(
                    current_load,
                    current_soc,
                    row['charge_limits'],
                    row['discharge_limits'],
                    row['cop_charge'],
                    row['cop_discharge'],
                    highest_threshold[period])

                # If not in testing window
                if (ts.time() < time_start_corrected or ts.time() > time_end):
                    # Disable charging outside active window if tank half full
                    if offset < 0 and current_soc > self.lt_capacity/2:
                        if row['target'] > highest_threshold[period]:
                            highest_threshold[period] = row['target']
                        continue


                day.at[index, 'soc'] = soc
                day.at[index, 'offsets'] = offset
                target = current_load - offset

                day.at[index, 'target'] = target

                # If target breaches threshold then set new highest threshold
                if target > highest_threshold[period]:
                    highest_threshold[period] = target

                current_soc = soc

            new_day_data.append(day)
        new_data = pd.concat(new_day_data)
        return new_data

    def apply_heat_leak(self, current_soc, heat_leak):
        """

        :param current_soc:
        :param heat_leak:
        :return:
        """
        next_soc = current_soc - (heat_leak * current_soc / 4.0)
        if next_soc < 0:
            next_soc = 0
        return next_soc

    def get_next_soc_offset(self, current_load, current_soc, chg_limit, dchg_limit,
                            chg_cop, dchg_cop, threshold):
        """

        :param current_load:
        :param current_soc:
        :param chg_limit:
        :param dchg_limit:
        :param chg_cop:
        :param dchg_cop:
        :param threshold:
        :return:
        """
        threshold_offset = current_load - threshold

        if threshold_offset == 0:  # DMT mode
            offset = 0
            soc = current_soc
        elif threshold_offset < 0:  # CHG mode
            offset = max(threshold_offset, -chg_limit)
            soc = current_soc - (offset * chg_cop / 4.0)
            if soc > self.lt_capacity:
                offset = (current_soc - self.lt_capacity) / chg_cop * 4.0
                soc = self.lt_capacity
        else:  # DCHG mode
            offset = min(threshold_offset, dchg_limit)
            soc = current_soc - (offset * dchg_cop / 4.0)
            if soc < self.tank_min:
                offset = (current_soc - self.tank_min) / dchg_cop * 4.0
                soc = self.tank_min
        return soc, offset
