import matplotlib.pyplot as plt


def baseline_plot(x_data, y_data, prediction):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(x_data['timestamp'], y_data['building_baseline'], label='Baseline')
    ax.axhline(prediction, color='red')
    ax.set_title(f'{x_data.iloc[0]["date_site"]}')
    plt.show()


def lt_plot(site, data, t1=None, t2=None, max_peak=None):
    if t1 is None:
        t1 = data.index[0]
    if t2 is None:
        t2 = data.index[-1]

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.title(f"LT Optimization for {site}")

    # plotting power related graphs on first y axis
    ln1 = ax.plot(data.offsets[t1:t2],
                  label="Offset")
    ln2 = ax.plot(data.baseline[t1:t2],
                  label='Building Power')
    ln3 = ax.plot(data.load_values[t1:t2],
                  label='New Building Power')
    ln4 = ax.plot(-data.charge_limits[t1:t2],
                  label='Max Charge Values')
    ln5 = ax.plot(data.discharge_limits[t1:t2],
                  label='Max Discharge Values')
    ln7 = ax.plot(data.temperature[t1:t2],
                  label='Temperature')

    ax.set_xlabel('Time period')
    ax.set_ylabel('Power in KW')

    start, end = ax.get_xlim()
    # ax.set_xticks(np.arange(start, end, 8))
    # ax.set_xticklabels(sv)
    # plt.xticks(np.arange(start+33.55, end+33.55, 16), sv, rotation=90);

    # creating second y axis and plotting SOC
    ax2 = ax.twinx()
    ln6 = ax2.plot(data.soc[t1:t2], '--',
                   label="SOC")

    ax2.set_ylabel('State of charge in KWhe')

    ax.axhline(0, color='black')

    # Managing labels and legend
    lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6 + ln7
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    if max_peak:
        ax2.set_ylim(-14, max_peak)
    return ax
