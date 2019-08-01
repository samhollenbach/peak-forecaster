import pytz
import pandas as pd
import numpy as np
from functools import partial
import os
import csv
from peak_forecaster import thermal_util
from pickle_jar.pickle_jar import pickle_jar

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


@pickle_jar(detect_changes=True, reload=True)
def load_data(site, power_file, site_info, start=None, end=None, thermal_info=True):
    power_data = read_power_data(site, power_file, start, end)
    power_data['mbh'] = float(site_info['lt_mbh']) + float(site_info['mt_mbh'])
    if thermal_info:
        master_conf = get_thermal_config(site, start, end,
                                         '../input/WM_LTSB_mass_and_SST_new.csv')
        master_conf[
            'MRC'] = power_data['crs_baseline'].max() + 2
        power_data = thermal_util.add_thermal_info(power_data, master_conf)
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
            site_data = load_data(site, file, site_info[site])
            site_data['site'] = site
            data.append(site_data)
    else:
        for site, file in zip(sites, files):
            data.append(load_data(site, file, site_info[site]))
    data = pd.concat(data, sort=False)
    data.reset_index(inplace=True)
    return data

def get_thermal_config(site, start, end, lt_config_file):
    if site == 'WFROS' or site == 'WFROS':
        site_id = 'pge_e19_2019'
    else:
        site_id = 'pge_e19_2019'

    lt_conf = get_lt_config(lt_config_file)
    lt_capacity = lt_conf.loc[lt_conf.Store == site]['mass_derated'].iloc[0]
    sst_max_f = lt_conf.loc[lt_conf.Store == site]['SST_max'].iloc[0]
    sst_min_f = lt_conf.loc[lt_conf.Store == site]['SST_min'].iloc[0]
    sst_mid_f = (sst_max_f + sst_min_f) / 2
    cop_mid_sst = partial(thermal_util.master_cop_eq, thermal_util.farenheit_to_celsius(sst_mid_f))
    cop_max_sst = partial(thermal_util.master_cop_eq, thermal_util.farenheit_to_celsius(sst_max_f))
    master = {
        'site': site,
        'site_id': site_id,
        'start': start,
        'end': end,
        'lt_config': lt_conf.loc[lt_conf.Store == site].to_dict(),
        'lt_capacity': lt_capacity,
        'sst_max_f': sst_max_f,
        'sst_mid_f': sst_mid_f,
        'sst_min_f': sst_min_f,
        'cop_mid_sst': cop_mid_sst,
        'cop_max_sst': cop_max_sst,
        'sst_factor': 0.15,
    }
    return master

def get_lt_config(config_file):
    config_lt = pd.read_csv(config_file)
    return config_lt
