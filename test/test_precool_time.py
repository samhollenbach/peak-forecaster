import os
import pickle
from peak_forecaster.display import show
from peak_forecaster import savings
import matplotlib.pyplot as plt
import datetime

site = 'WM3140_optimizer'
path = f'output/{site}'
files = os.listdir(path)
targets = []

for file in files:
    with open(os.path.join(path, file), 'rb') as r:
        d = pickle.load(r)
        targets.append(d)

        # s = savings.Savings('pge_e19_2019')
        # total_savings = s.get_savings(d[['timestamp', 'building_baseline']], d[['timestamp', 'target']])
        # baseline_plot2(d, title=f'{site} - Savings: ${total_savings}')


start_times = []
end_times = []
sum_charged = []
sum_discharged = []
max_peaks = []
peak_offset = []

for target in targets:

    target['date'] = target['timestamp'].apply(lambda t: t.date())
    grouped = [d for _, d in target.groupby('date')]


    for day_target in grouped:

        start = None
        end = None

        charge_sum = 0
        discharge_sum = 0

        max_offset = 0
        max_peak = 0

        for index, row in day_target.iterrows():

            ts = row['timestamp']
            offset = row['offsets']
            baseline = row['building_baseline']

            if baseline > max_peak:
                max_peak = baseline
                max_offset = offset

            if offset != 0 and start is None:
                start = ts
                # print(f"Start: {start}")

            if offset != 0:
                end = ts


            if offset < 0:
                charge_sum += offset
            elif offset > 0:
                discharge_sum += offset

        if start is None:
            continue

        start_times.append(start)
        end_times.append(end)
        sum_charged.append(charge_sum / 4)
        sum_discharged.append(discharge_sum / 4)
        max_peaks.append(max_peak)
        peak_offset.append(max_offset)

    print(max_peaks)
    print(start_times)
    print(end_times)
    print(sum_charged)
    print(sum_discharged)

start_t = [s.time() for s in start_times]
end_t = [e.time() for e in end_times]

# plt.scatter(start_t, max_peaks)
# plt.xlim(datetime.time(0, 0), datetime.time(23, 59))
# plt.xlabel('Start Time of Charge')
# plt.ylabel('Maximum Baseline Peak (kW)')
# plt.title('WM3140 - Optimizer')
# plt.show()
#
#
# plt.scatter(end_t, max_peaks)
# plt.xlim(datetime.time(0, 0), datetime.time(23, 59))
# plt.xlabel('End Time of Discharge')
# plt.ylabel('Maximum Baseline Peak (kW)')
# plt.title('WM3140 - Optimizer')
# plt.show()

# plt.scatter(start_t, peak_offset)
# plt.xlim(datetime.time(0, 0), datetime.time(23, 59))
# plt.xlabel('Start Time of Charge')
# plt.ylabel('Offset Commanded at Peak (kW)')
# plt.title('WM3140 - Optimizer')
# plt.show()
#
#
# plt.scatter(end_t, peak_offset)
# plt.xlim(datetime.time(0, 0), datetime.time(23, 59))
# plt.xlabel('End Time of Discharge')
# plt.ylabel('Offset Commanded at Peak (kW)')
# plt.title('WM3140 - Optimizer')
# plt.show()


#
# plt.scatter(sum_discharged, peak_offset, c=max_peaks)
# plt.xlabel('Amount Discharged During Day  (kWh)')
# plt.ylabel('Offset Commanded at Peak (kW)')
# plt.title('WM3140 - Optimizer')
# plt.show()
