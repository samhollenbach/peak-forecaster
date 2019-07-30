from pickle_jar.pickle_jar import pickle_jar
import numpy as np
import pandas as pd


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

            x_stats = x_data.describe()


            # Add encoded month
            month_scaled = int(x_data.iloc[0]['month']) / 12.0
            # month_components = (np.sin(month_scaled), np.cos(month_scaled))
            # month_components = (np.sin(month_scaled),)
            day_x.append(month_scaled)

            # Add encoded day of week
            day_of_week = x_data.iloc[0]['day_of_week']
            day_of_week_int = days_of_week.index(day_of_week) / 7.0
            day_x.append(day_of_week_int)

            # Add mbh
            mbh = x_data.iloc[0]['mbh']
            day_x.append(mbh)

            # Add temperature stats
            day_x.append(x_stats.at['max', 'temperature'])
            day_x.append(x_stats.at['min', 'temperature'])
            day_x.append(x_stats.at['mean', 'temperature'])
            # print(day_features)
            features_x.append(day_x)

            # Find baseline peak load
            y_data_copy = y_data.copy()
            peak_load = y_data_copy['building_baseline'].loc[y_data_copy['building_baseline'].idxmax()]
            # y_data = pd.DataFrame(y_data)
            # interval = y_data.loc[y_data['building_baseline'].idxmax()]
            # peak_dt = pd.to_datetime(interval['timestamp'])
            # peak_time_scaled = (peak_dt.hour * 60 + peak_dt.minute) / (24 * 60)
            # peak_components = (np.sin(peak), np.cos(peak))
            # peak_components = (np.sin(peak),)
            # day_y.append(peak_time_scaled)
            day_y.append(peak_load)
            features_y.append(day_y)
        return features_x, features_y

    all_train_data = pd.concat(x_train)
    stats = all_train_data.describe()

    features_x_train, features_y_train = get_stats(x_train, y_train, stats)
    features_x_test, features_y_test = get_stats(x_test, y_test, stats)

    return np.array(features_x_train), np.array(features_y_train),\
           np.array(features_x_test), np.array(features_y_test)