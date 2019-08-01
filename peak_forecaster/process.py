from pickle_jar.pickle_jar import pickle_jar
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, time
import pytz
from collections import deque

days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
encoding_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
day_of_week_encoding = {}
for i, day in enumerate(days_of_week):
    encoding = encoding_base.copy()
    encoding[i] = 1.0
    day_of_week_encoding[day] = tuple(encoding)


def split_x_y(data):
    x = []
    y = []
    for day in data:
        x.append(day.copy().drop('building_baseline', axis=1))
        y.append(day[['building_baseline', 'timestamp']])
    return x, y


# @pickle_jar(reload=False)
def train_test_split(data, size=0.2, seed=None, test_start_date=None, test_end_date=None,
                     timestamp_column='timestamp'):
    # Output lists
    train = []
    test = []

    # Group data by day
    daily_data = [day for _, day in data.groupby('date_site')]

    # Random seed
    np.random.seed(seed)
    rands = np.random.rand(len(daily_data))

    # Lag variables
    num_lags = 5
    lag_vals = deque([])
    last_site = None

    for i, day in enumerate(daily_data):

        # Assign lags
        max_peak = day['building_baseline'].max()
        if day.iloc[0]['site'] != last_site:
            last_site = day.iloc[0]['site']
            lag_vals = deque([max_peak])
            continue
        if len(lag_vals) < num_lags:
            lag_vals.append(max_peak)
            continue
        else:
            for lag_num, val in enumerate(lag_vals):
                day[f'lag_{lag_num}'] = val
            _ = lag_vals.popleft()
            lag_vals.append(max_peak)

        # Assign test or train
        if test_start_date is None and test_end_date is None:
            if rands[i] <= size:
                test.append(day)
            else:
                train.append(day)
        else:
            date = day.iloc[0][timestamp_column].date()
            if date >= test_start_date and date <= test_end_date:
                test.append(day)
            else:
                train.append(day)

    return train, test


def extract_features(data):

    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    encoding_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    day_of_week_encoding = {}
    for i, day in enumerate(days_of_week):
        encoding = encoding_base.copy()
        encoding[i] = 1.0
        day_of_week_encoding[day] = tuple(encoding)

    def get_stats(data):
        features_x = []
        features_y = []

        x, y = split_x_y(data)

        for x_data, y_data in zip(x, y):
            day_x = []
            day_y = []

            y_data_frame = y_data.copy()
            y_data_frame = pd.DataFrame(y_data_frame)

            # Find baseline peak load
            peak_load = y_data['building_baseline'].loc[
                y_data['building_baseline'].idxmax()]
            interval = y_data_frame.loc[
                y_data_frame['building_baseline'].idxmax()]
            peak_dt = pd.to_datetime(interval['timestamp'])
            date = peak_dt.date()
            peak_time_minute = (peak_dt.hour * 60 + peak_dt.minute)
            day_y.append(peak_load)
            # day_y.append(peak_time_minute)

            # Determine if day has solar
            solar_start = datetime.combine(date, time(12, 0,
                                                      tzinfo=pytz.timezone(
                                                          'US/Pacific')))
            solar_end = datetime.combine(date, time(14, 0,
                                                    tzinfo=pytz.timezone(
                                                        'US/Pacific')))
            reg_start = datetime.combine(date, time(0, 0, tzinfo=pytz.timezone(
                'US/Pacific')))
            reg_end = datetime.combine(date, time(8, 0, tzinfo=pytz.timezone(
                'US/Pacific')))
            dts = y_data.copy()
            solar_data = dts.loc[(dts['timestamp'] > solar_start) & (
                        dts['timestamp'] < solar_end)]
            reg_data = dts.loc[
                (dts['timestamp'] > reg_start) & (dts['timestamp'] < reg_end)]
            solar_avg = np.average(solar_data['building_baseline'])
            reg_avg = np.average(reg_data['building_baseline'])
            solar = 1 if solar_avg < reg_avg else 0
            day_x.append(solar)

            x_stats = x_data.describe()

            # Add is holiday
            cal = USFederalHolidayCalendar()
            holiday = cal.holidays(start=date, end=date)
            h = 0 if holiday.empty else 1
            day_x.append(h)

            # Add encoded month
            month_scaled = int(x_data.iloc[0]['month']) / 12.0
            month_components = (
            np.sin(np.pi * month_scaled), np.cos(np.pi * month_scaled))
            # month_components = (np.sin(month_scaled),)
            day_x.extend(month_components)

            # Add encoded day of week
            dow = x_data.iloc[0]['day_of_week']
            day_x.extend(day_of_week_encoding[dow])

            # Add mbh
            mbh = x_data.iloc[0]['mbh']
            day_x.append(mbh)

            # Add temperature stats
            temp_fudge = 0

            day_x.append(x_stats.at['max', 'temperature'] + temp_fudge)
            day_x.append(x_stats.at['min', 'temperature'] + temp_fudge)
            day_x.append(x_stats.at['mean', 'temperature'] + temp_fudge)

            nlags = 5
            for i in range(nlags):
                day_x.append(x_data.iloc[0][f'lag_{i}'])

            features_x.append(day_x)
            features_y.append(day_y)
        return features_x, features_y

    # all_train_data = pd.concat(x_train)
    # stats = all_train_data.describe()

    print("Extracting Train Set Features...\n")
    features_x, features_y = get_stats(data)

    return np.array(features_x), np.array(features_y)


# @pickle_jar(reload=True)
# def extract_features(x_train, y_train, x_test, y_test):
#
#
#
#     def get_stats(x, y):
#         features_x = []
#         features_y = []
#         for x_data, y_data in zip(x, y):
#             day_x = []
#             day_y = []
#
#
#             y_data_frame = y_data.copy()
#             y_data_frame = pd.DataFrame(y_data_frame)
#
#             # Find baseline peak load
#             peak_load = y_data['building_baseline'].loc[y_data['building_baseline'].idxmax()]
#             interval = y_data_frame.loc[y_data_frame['building_baseline'].idxmax()]
#             peak_dt = pd.to_datetime(interval['timestamp'])
#             date = peak_dt.date()
#             peak_time_minute = (peak_dt.hour * 60 + peak_dt.minute)
#             day_y.append(peak_load)
#             # day_y.append(peak_time_minute)
#
#
#             # Determine if day has solar
#             solar_start = datetime.combine(date, time(12, 0, tzinfo=pytz.timezone('US/Pacific')))
#             solar_end = datetime.combine(date, time(14, 0, tzinfo=pytz.timezone('US/Pacific')))
#             reg_start = datetime.combine(date, time(0, 0, tzinfo=pytz.timezone('US/Pacific')))
#             reg_end = datetime.combine(date, time(8, 0, tzinfo=pytz.timezone('US/Pacific')))
#             dts = y_data.copy()
#             solar_data = dts.loc[(dts['timestamp'] > solar_start) & (dts['timestamp'] < solar_end)]
#             reg_data = dts.loc[(dts['timestamp'] > reg_start) & (dts['timestamp'] < reg_end)]
#             solar_avg = np.average(solar_data['building_baseline'])
#             reg_avg = np.average(reg_data['building_baseline'])
#             solar = 1 if solar_avg < reg_avg else 0
#             day_x.append(solar)
#
#             x_stats = x_data.describe()
#
#             # Add is holiday
#             cal = USFederalHolidayCalendar()
#             holiday = cal.holidays(start=date, end=date)
#             h = 0 if holiday.empty else 1
#             day_x.append(h)
#
#             # Add encoded month
#             month_scaled = int(x_data.iloc[0]['month']) / 12.0
#             month_components = (np.sin(np.pi * month_scaled), np.cos(np.pi * month_scaled))
#             # month_components = (np.sin(month_scaled),)
#             day_x.extend(month_components)
#
#             # Add encoded day of week
#             dow = x_data.iloc[0]['day_of_week']
#             day_x.extend(day_of_week_encoding[dow])
#
#             # Add mbh
#             mbh = x_data.iloc[0]['mbh']
#             day_x.append(mbh)
#
#             # Add temperature stats
#             temp_fudge = 0
#
#             day_x.append(x_stats.at['max', 'temperature'] + temp_fudge)
#             day_x.append(x_stats.at['min', 'temperature'] + temp_fudge)
#             day_x.append(x_stats.at['mean', 'temperature'] + temp_fudge)
#
#             nlags = 5
#             for i in range(nlags):
#                 day_x.append(x_data.iloc[0][f'lag_{i}'])
#
#
#             features_x.append(day_x)
#             features_y.append(day_y)
#         return features_x, features_y
#
#     # all_train_data = pd.concat(x_train)
#     # stats = all_train_data.describe()
#
#     print("Extracting Train Set Features...\n")
#     features_x_train, features_y_train = get_stats(x_train, y_train)
#     print("Extracting Test Set Features...\n")
#     features_x_test, features_y_test = get_stats(x_test, y_test)
#
#     return np.array(features_x_train), np.array(features_y_train),\
#            np.array(features_x_test), np.array(features_y_test)

