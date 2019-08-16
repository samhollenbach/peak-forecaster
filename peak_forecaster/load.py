import pytz
import pandas as pd
import pickle
import numpy as np
from functools import partial
import os
import csv
from peak_forecaster import thermal_util
from tariff import Tariff
from pickle_jar.pickle_jar import pickle_jar

def read_power_data(site, power_file, start, end):

    target_tz = pytz.timezone('US/Pacific')

    try:
        with open(power_file, 'rb') as r:
            df = pickle.load(r)
    except:
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


    df.set_index('timestamp', inplace=True)

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

    t = Tariff('pge')
    df = t.apply_period(df, 'timestamp')


    if 'temperature' not in df.columns and 'dbt' in df.columns:
        df['temperature'] = df['dbt'].copy()
        df.drop('dbt', axis=1, inplace=True)

    if df['temperature'].max() < 60:
        df['temperature'] =  df['temperature'].apply(thermal_util.celsius_to_fahrenheit)

    if 'crs_new' in df.columns:
        df['crs_baseline'] = df['crs_new'].copy()
        df.drop('crs_new', axis=1, inplace=True)

    if 'crs.baseline.power.kW' in df.columns:
        df['crs_baseline'] = df['crs.baseline.power.kW'].copy()
        df.drop('crs.baseline.power.kW', axis=1, inplace=True)

    df['crs_baseline'] = df['crs_baseline'].interpolate('linear')
    df['building_baseline'] = df['building_baseline'].interpolate('linear')
    df['temperature'] = df['temperature'].interpolate('linear')

    df['baseline_no_crs'] = df['building_baseline'] - df['crs_baseline']

    df.reset_index(drop=True, inplace=True)

    print(df['timestamp'].head())

    return df


def get_site_info():
    site_info = {}
    with open('../input/site_info.csv', 'r') as r:
        reader = csv.DictReader(r)
        for row in reader:
            site_info[row['site']] = {k: v for k, v in row.items() if
                                      k != 'site'}
    return site_info

# @pickle_jar(detect_changes=True, reload=True)
def load_data(site, site_info, power_file=None, start=None, end=None, thermal_info=True):
    if power_file is None:
        path = '../input/'
        power_file = [os.path.join(path, i) for i in os.listdir(path) if
                os.path.isfile(os.path.join(path, i)) and
                site in i][0]

    power_data = read_power_data(site, power_file, start, end)
    power_data['mbh'] = float(site_info['lt_mbh']) + float(site_info['mt_mbh'])
    if thermal_info:


        if site.startswith('WF'):
            config_file = '../input/WF_LTSB_mass_and_SST.csv'
        else:
            config_file = '../input/WM_LTSB_mass_and_SST_new.csv'

        master_conf = get_thermal_config(site, start, end,
                                         config_file)
        master_conf[
            'MRC'] = power_data['crs_baseline'].max() + 2

        master_conf['optimizer_config']['MRC'] = master_conf['MRC']
        power_data = thermal_util.add_thermal_info(power_data, master_conf)
        return power_data, master_conf
    return power_data

# @pickle_jar(reload=True)
def load_all_data(sites, files=None):
    data = []

    site_info = get_site_info()

    if files is None:
        for site in sites:

            site_data, _ = load_data(site, site_info[site])
            site_data['site'] = site
            data.append(site_data)
    else:
        for site, file in zip(sites, files):
            dat, _ = load_data(site, site_info[site], power_file=file)
            data.append(dat)
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
    cop_mid_sst = partial(thermal_util.master_cop_eq, thermal_util.fahrenheit_to_celsius(sst_mid_f))
    cop_max_sst = partial(thermal_util.master_cop_eq, thermal_util.fahrenheit_to_celsius(sst_max_f))
    master = {
        'site': site,
        'site_id': site_id,
        'start': start,
        'end': end,
        'optimizer_config': get_optimizer_config(site_id, start, end,
                                                 lt_capacity),
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


def get_optimizer_config(site_id, start, end, lt_capacity):
    config = {
        "timezone": "US/Pacific",
        "site_id": site_id,
        "start": start,
        "end": end,
        "MDL": 1000,
        "MCL": 1000,
        "min_charge_offset": 5,
        "min_discharge_offset": 5,
        # "RTE_setpoint": 0.65,
        "RB_capacity": lt_capacity,
        "M": 1000,
        "SOC_initial": 0,
        "cop_dchg_coefficients": [0],  # Already provided in timeseries data
        "cop_chg_coefficients": [0],  # Already provided in timeseries data
        "constraints": {
            "time_transition": False,
            "minimum_charge_offset": False,
            "minimum_discharge_offset": False,
            "chg_limit_curve": False,
            "dchg_limit_curve": False,
            "fixed_rte": False
        },
        "outputs": {
            "timestamp": True,
            "baseline": True,
            "offsets": True,
            "load_values": True,
            "soc": True,
            "charge_limits": True,
            "discharge_limits": True,
            "cop_dchg": True,
            "cop_chg": True,
            "temperature": True
        }
    }
    return config

def get_lt_config(config_file):
    config_lt = pd.read_csv(config_file)
    return config_lt
