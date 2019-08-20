from test_operator import run_operator

site = 'WFNPA'
perfect_forecast = True


year = '2018'
months = range(5, 11)
days = ['01', '07', '15', '22']
time_ranges = [(f'{year}-{m:02d}-{d} 00:00:00-07', f'{year}-{m+1:02d}-{d} 23:45:00-07') for m in months for d in days]



#######################################
# Set Up Derating Factors Grid Search #
#######################################
# TODO: Move this out of here and clean up into function
# nc_range = (0.45, 0.45)
# onp_range = (0.75, 0.75)
# mp_range = (0.3, 1.0)
# step = 0.05
# nc_derates = np.linspace(*nc_range, round((nc_range[1]-nc_range[0])/step)+1)
# onp_derates = np.linspace(*onp_range, round((onp_range[1]-onp_range[0])/step)+1)
# mp_derates = np.linspace(*mp_range, round((mp_range[1]-mp_range[0])/step)+1)
#
# derating_factors = [{ 'Non-Coincident': d[0], 'On-Peak': d[1], 'Mid-Peak': d[2]}
#                     for d in itertools.product(nc_derates, onp_derates, mp_derates)]
#
# for der_facs in [{'Non-Coincident': 0.4, 'On-Peak': 0.7}]:


# Use these derating factors
der_facs = {'Non-Coincident': 0.45}

#################
# Operator Runs #
#################

# -1 = Use all CPU's
processes = -1

# Do run for each time range in time_ranges
results = run_operator(site, time_ranges,
                       derating_factors=der_facs,
                       processes=processes,
                       perfect_forecast=perfect_forecast,
                       solar=False,
                       prediction_periods=('Non-Coincident',))




#####################
# Aggregate Results #
#####################
# saving_results_fields = ('Derating Factors', 'Demand Non-Coincident', 'Demand On-Peak', 'Demand Mid-Peak', 'Demand', 'Energy')

savings_total = []
savings_demand = []
savings_demand_nc = []
savings_demand_onp = []
savings_demand_midp = []
savings_energy = []

for targets, savings_breakdown in results:
    # pprint(savings_breakdown)
    dncs = savings_breakdown['Demand Non-Coincident']['Savings']
    donps = savings_breakdown['Demand On-Peak']['Savings']
    dmps = savings_breakdown['Demand Mid-Peak']['Savings']
    ds = dncs + donps + dmps
    es = savings_breakdown['Energy Use']['Savings']
    ts = ds + es
    savings_total.append(ts)
    savings_demand.append(ds)
    savings_demand_nc.append(dncs)
    savings_demand_onp.append(donps)
    savings_demand_midp.append(dmps)
    savings_energy.append(es)

    # Display plot of results
    # display.show(targets, savings=(ts, ds, es))

results = (
    der_facs,
    f"Demand Non-Coincident: ${sum(savings_demand_nc)/len(savings_demand_nc):0.2f}",
    f"Demand On-Peak: ${sum(savings_demand_onp)/len(savings_demand_onp):0.2f}",
    f"Demand Mid-Peak: ${sum(savings_demand_midp)/len(savings_demand_midp):0.2f}",
    f"Demand: ${sum(savings_demand) / len(savings_demand):0.2f}",
    f"Energy: ${sum(savings_energy) / len(savings_energy):0.2f}",
    f"Net: ${sum(savings_total) / len(savings_total):0.2f}",
    )

# savings_results = []
# savings_results.append(results)

print("Summer Month Runs")
print(f"Net: {savings_total}")
print(f"Demand: {savings_demand}")
print(f"Energy: {savings_energy}")
print("\nAverage Savings")
for res in results:
    print(res)
print("\nFinished...\n\n")


# pprint(savings_results)
