from peak_forecaster import load, iterative_optimizer, display, savings
from savings import SavingsCalculator
from pprint import pprint
import pandas as pd
import os
from tariff import Tariff


site = 'WFLAT'

runs = 0
savings_total = []
savings_demand = []
savings_energy = []

if site == 'WFLAT':
    year = '2019'
    months = range(5, 8)
else:
    year = '2018'
    months = range(5, 11)
days = ['01', '07', '15', '22']
time_ranges = [(f'{year}-{m:02d}-{d} 00:00:00-07',
                f'{year}-{m+1:02d}-{d} 23:45:00-07') for m in months for d in days]

if site == 'WFLAT':
    time_ranges = time_ranges[3:-2]

for start, end in sorted(time_ranges):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    site_info = load.get_site_info()[site]
    data, config = load.load_data(site, site_info, start=start_dt, end=end_dt, thermal_info=True)

    config['optimize_energy'] = False

    # print(data.head())
    # pprint(config)



    targets = iterative_optimizer.run_iterative_optimizer([data], config, verbose=True)
    targets = targets[0]
    targets['timestamp'] = data.copy()['timestamp']
    t = Tariff('wfla')
    targets = t.apply_period(targets)

    targets = targets.rename(columns={
        'baseline': 'building_baseline',
        'load_values': 'target',
    })

    targets['threshold'] = targets['target'].max()

    s = savings.Savings(config['site_id'])


    overrides = {
        'baseline': 'building_baseline',
        'actual': 'target',
    }

    sc = SavingsCalculator('wfla', mapping_overrides=overrides)
    breakdown = sc.calculate_savings_breakdown(targets, None,
                                               group_over='season',
                                               total_energy=True,
                                               breakdown_energy=False)
    breakdown = breakdown['actual_savings_breakdown']['all']
    pprint(breakdown)

    # Calculate operational savings

    demand_savings = breakdown['Demand Non-Coincident']['Savings'] \
                     + breakdown['Demand Mid-Peak']['Savings']\
                     + breakdown['Demand On-Peak']['Savings']
    energy_savings = breakdown['Energy Use']['Savings']

    total_savings = demand_savings + energy_savings
    print(f"Savings: ${total_savings:.2f}")
    print(f"Demand Savings: ${demand_savings:.2f}")
    print(f"Energy Savings: ${energy_savings:.2f}")



    savings_total.append(total_savings)
    savings_demand.append(demand_savings)
    savings_energy.append(energy_savings)

    if not os.path.exists(f'output/{site}_optimizer/'):
        os.mkdir(f'output/{site}_optimizer/')

    targets.to_pickle(
        f'output/{site}_optimizer/LT_Setback_EBM_{site}_'
        f'{start.split(" ")[0]}_{end.split(" ")[0]}.pickle')



    title = f"{site} - {start.split(' ')[0]}"
    display.show(targets, title=title, savings=[total_savings, demand_savings, energy_savings])
    runs += 1


print("All Runs")
print("Total:", savings_total)
print("Demand:", savings_demand)
print("Energy:", savings_energy)
print("Average")
print("Total:", sum(savings_total)/runs)
print("Demand:", sum(savings_demand)/runs)
print("Energy:", sum(savings_energy)/runs)