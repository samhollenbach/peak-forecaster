from pickle_jar.pickle_jar import pickle_jar
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, time
import pytz

def time_from_component(time_scaled):
    ts = time_scaled * (24 * 60)
    hour = np.floor(ts / 60)
    min = ts - (hour * 60)
    return hour, min


@pickle_jar(reload=True)
def extract_features(x_train, y_train, x_test, y_test):
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def get_stats(x, y, stats):
        features_x = []
        features_y = []
        for x_data, y_data in zip(x, y):
            day_x = []
            day_y = []


            y_data_frame = y_data.copy()
            y_data_frame = pd.DataFrame(y_data_frame)

            # Find baseline peak load
            peak_load = y_data['building_baseline'].loc[y_data['building_baseline'].idxmax()]
            interval = y_data_frame.loc[y_data_frame['building_baseline'].idxmax()]
            peak_dt = pd.to_datetime(interval['timestamp'])
            date = peak_dt.date()
            peak_time_minute = (peak_dt.hour * 60 + peak_dt.minute)
            day_y.append(peak_load)
            day_y.append(peak_time_minute)


            # Determine if day has solar
            solar_start = datetime.combine(date, time(12, 0, tzinfo=pytz.timezone('US/Pacific')))
            solar_end = datetime.combine(date, time(14, 0, tzinfo=pytz.timezone('US/Pacific')))
            # solar_range = pd.date_range(solar_start, solar_end)
            reg_start = datetime.combine(date, time(0, 0, tzinfo=pytz.timezone('US/Pacific')))
            reg_end = datetime.combine(date, time(8, 0, tzinfo=pytz.timezone('US/Pacific')))
            # reg_range = pd.date_range(reg_start, reg_end)

            dts = y_data.copy()
            solar_data = dts.loc[(dts['timestamp'] > solar_start) & (dts['timestamp'] < solar_end)]
            reg_data = dts.loc[(dts['timestamp'] > reg_start) & (dts['timestamp'] < reg_end)]

            solar_avg = np.average(solar_data['building_baseline'])
            reg_avg = np.average(reg_data['building_baseline'])

            solar = 0
            if solar_avg < reg_avg:
                solar = 1
            day_x.append(solar)

            x_stats = x_data.describe()

            # Add is holiday
            cal = USFederalHolidayCalendar()
            holiday = cal.holidays(start=date, end=date)
            h = 0 if holiday.empty else 1
            day_x.append(h)

            # Add encoded month
            month_scaled = int(x_data.iloc[0]['month']) / 12.0
            month_components = (np.sin(np.pi * month_scaled), np.cos(np.pi * month_scaled))
            # month_components = (np.sin(month_scaled),)
            day_x.extend(month_components)

            # Add encoded day of week
            day_of_week = x_data.iloc[0]['day_of_week']
            day_of_week_int = days_of_week.index(day_of_week) / 7.0
            day_x.append(day_of_week_int)

            # Add mbh
            mbh = x_data.iloc[0]['mbh']
            day_x.append(mbh)

            # Add temperature stats
            temp_fudge = 0

            day_x.append(x_stats.at['max', 'temperature'] + temp_fudge)
            day_x.append(x_stats.at['min', 'temperature'] + temp_fudge)
            day_x.append(x_stats.at['mean', 'temperature'] + temp_fudge)
            # print(day_features)



            features_x.append(day_x)
            features_y.append(day_y)
        return features_x, features_y

    all_train_data = pd.concat(x_train)
    stats = all_train_data.describe()

    features_x_train, features_y_train = get_stats(x_train, y_train, stats)
    features_x_test, features_y_test = get_stats(x_test, y_test, stats)

    return np.array(features_x_train), np.array(features_y_train),\
           np.array(features_x_test), np.array(features_y_test)