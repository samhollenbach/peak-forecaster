import numpy as np
from tariff.bill_calculator import BillCalculator
import pandas as pd

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
        self.lt_capacity = 3600
        self.tank_min = 0
        self.bill_calculator = BillCalculator('pge')

    def get_savings(self, baseline_load, target_load):
        base_bill = self.bill_calculator.calculate_total_bill(baseline_load, load_column='building_baseline')
        target_bill = self.bill_calculator.calculate_total_bill(target_load, load_column='target')
        return base_bill - target_bill

    def prepare_data(self, data):
        data = data.copy()
        data = data[
            (data.timestamp >= self.start) & (data.timestamp <= self.end)]
        data['offsets'] = 0
        data['soc'] = 0
        data['target'] = data['building_baseline']

        data = data.assign(
            date=data.timestamp.apply(
                lambda t: f"{t:%Y}-{t:%m}-{t:%d}"))
        daily_data = [day_data for _, day_data in data.groupby('date_site')]
        return daily_data


    def run_standard_operation(self, time_start, time_end, data, thresholds):
        highest_threshold = thresholds[0]

        daily_data = self.prepare_data(data)

        if len(daily_data) != len(thresholds):
            raise ValueError("Number of days in data and amount of thresholds do not match")


        new_day_data = []
        for i, day in enumerate(daily_data):
            current_soc = 0

            day['threshold_old'] = thresholds[i]

            if thresholds[i] > highest_threshold:
                highest_threshold = thresholds[i]

            day['threshold'] = highest_threshold
            for index, row in day.iterrows():
                ts = row['timestamp']
                day.at[index, 'threshold'] = highest_threshold
                if ts.time() < time_start or ts.time() > time_end:
                    continue

                current_load = row['building_baseline']
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

        if threshold_offset == 0:  # DMT mode
            offset = 0
            soc = current_soc
        elif threshold_offset < 0:  # CHG mode
            offset = max(threshold_offset, -chg_limit)
            soc = current_soc - (offset * chg_cop)
            if soc > self.lt_capacity:
                offset = (self.lt_capacity - current_soc) / chg_cop
                soc = self.lt_capacity
        else:  # DCHG mode
            offset = min(threshold_offset, dchg_limit)
            soc = current_soc - (offset * dchg_cop)
            if soc < self.tank_min:
                offset = (self.tank_min - current_soc) / dchg_cop
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

