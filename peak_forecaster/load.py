import pytz
import pandas as pd
import numpy as np
from pickle_jar.pickle_jar import pickle_jar

import os
import csv

def read_power_data(site, power_file, start, end):

    target_tz = pytz.timezone('US/Pacific')


    try:
        df = pd.read_csv(power_file, parse_dates=['timestamp'])
    except ValueError:
        df = pd.read_excel(power_file, parse_dates=['timestamp'])

    df['timestamp'] = df['timestamp'].apply(
        lambda t: t.replace(tzinfo=target_tz))

    if start is None:
        start = df['timestamp'].iloc[0]
    else:
        try:
            start = start.tz_localize(target_tz)
        except TypeError:
            start = start.tz_convert(target_tz)
        if start < df.iloc[0]['timestamp']:
            print("Start time out of bounds of data, clipping")
            start = df.iloc[0]['timestamp']
    if end is None:
        end = df['timestamp'].iloc[-1]
    else:
        try:
            end = end.tz_localize(target_tz)
        except TypeError:
            end = end.tz_convert(target_tz)
        if end > df.iloc[-1]['timestamp']:
            print("End time out of bounds of data, clipping")
            end = df.iloc[-1]['timestamp']

    df = df[(df.timestamp >= start) & (df.timestamp <= end)]
    timestamps = pd.date_range(start, end, freq='15T', tz=target_tz)

    try:
        df['timestamp'] = timestamps
    except ValueError:
        df['timestamp'] = timestamps[:-4]
    df.set_index('timestamp', drop=True, inplace=True)

    df = df.assign(
        year_month=df.index.to_series(keep_tz=True).apply(
            lambda t: f"{t:%Y}-{t:%m}"))
    df = df.assign(
        month=df.index.to_series(keep_tz=True).apply(
            lambda t: f"{t:%m}"))
    df = df.assign(
        date=df.index.to_series(keep_tz=True).apply(
            lambda t: f"{t:%Y}-{t:%m}-{t:%d}"))

    df = df.assign(
        date_site=df.index.to_series(keep_tz=True).apply(
            lambda t: f"{site}-{t:%Y}-{t:%m}-{t:%d}"))

    df = df.assign(
        day_of_week=df.index.to_series(keep_tz=True).apply(
            lambda t: f"{t:%a}"))

    df['timestamp'] = df.index.to_series(keep_tz=True)

    if 'temperature' not in df.columns and 'dbt' in df.columns:
        df['temperature'] = df['dbt'].copy()
        df.drop('dbt', axis=1, inplace=True)

    df['temperature'] =  df['temperature'].apply(lambda t: (t - 32) * 5/9)

    if 'crs_new' in df.columns:
        df['crs_baseline'] = df['crs_new'].copy()
        df.drop('crs_new', axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


@pickle_jar(detect_changes=True, reload=False)
def load_data(site, power_file, site_info, start=None, end=None):
    power_data = read_power_data(site, power_file, start, end)
    power_data = power_data.drop('crs_baseline', axis=1)
    power_data['mbh'] = float(site_info['lt_mbh']) + float(site_info['mt_mbh'])
    return power_data

@pickle_jar(reload=True)
def load_all_data(sites, files=None):
    data = []

    site_info = {}

    with open('../input/site_info.csv', 'r') as r:
        reader = csv.DictReader(r)
        for row in reader:
            site_info[row['site']] = {k: v for k, v in row.items() if k != 'site'}

    if files is None:
        for site in sites:
            path = '../input/'
            file = [os.path.join(path, i) for i in os.listdir(path) if
                    os.path.isfile(os.path.join(path, i)) and
                    site in i][0]
            data.append(load_data(site, file, site_info[site]))
    else:
        for site, file in zip(sites, files):
            data.append(load_data(site, file, site_info[site]))
    data = pd.concat(data, sort=False)
    data.reset_index(inplace=True)
    return data

@pickle_jar(reload=True)
def train_test_split(data, size=0.8, seed=None):

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    day_data = [day for _, day in data.groupby('date_site')]

    np.random.seed(seed)
    rands = np.random.rand(len(day_data))

    for i, day in enumerate(day_data):
        y = day[['building_baseline', 'timestamp']]
        x = day.drop('building_baseline', axis=1)

        if rands[i] <= size:
            x_train.append(x)
            y_train.append(y)
        else:
            x_test.append(x)
            y_test.append(y)

    return x_train, y_train, x_test, y_test