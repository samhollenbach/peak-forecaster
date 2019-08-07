from peak_forecaster import load, iterative_optimizer, display, savings
from pprint import pprint
import pandas as pd

site = 'WM3140'

runs = 0
savings_total = []
savings_demand = []
savings_energy = []
for m in range(5, 11):

    for d in ['01', '07', '15', '22']:


        start = f'2018-{m:02d}-{d} 00:00:00-07'
        end = f'2018-{m + 1:02d}-{d} 23:45:00-07'
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

        targets = targets.rename(columns={
            'baseline': 'building_baseline',
            'load_values': 'target',
        })

        targets['threshold'] = targets['target'].max()

        s = savings.Savings(config['site_id'])
        # Calculate operational savings
        total_savings = s.get_savings(
            targets[['timestamp', 'building_baseline']],
            targets[['timestamp', 'target']])
        demand_savings = s.get_demand_savings(
            targets[['timestamp', 'building_baseline']],
            targets[['timestamp', 'target']])
        energy_savings = s.get_energy_savings(
            targets[['timestamp', 'building_baseline']],
            targets[['timestamp', 'target']])
        print(f"Savings: ${total_savings:.2f}")
        print(f"Demand Savings: ${demand_savings:.2f}")
        print(f"Energy Savings: ${energy_savings:.2f}")

        savings_total.append(total_savings)
        savings_demand.append(demand_savings)
        savings_energy.append(energy_savings)

        targets.to_pickle(
            f'output/{site}_optimizer/LT_Setback_EBM_{site}_'
            f'{start.split(" ")[0]}_{end.split(" ")[0]}.pickle')



        title = f"{site} - {start.split(' ')[0]}"
        # display.baseline_plot2(targets, title=title, savings=[total_savings, demand_savings, energy_savings])
        runs += 1


print("All Runs")
print("Total:", savings_total)
print("Demand:", savings_demand)
print("Energy:", savings_energy)
print("Average")
print("Total:", sum(savings_total)/runs)
print("Demand:", sum(savings_demand)/runs)
print("Energy:", sum(savings_energy)/runs)