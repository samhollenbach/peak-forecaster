import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['timezone'] = 'America/Los_Angeles'

def baseline_plot(x_data, y_data, prediction):
    data = x_data.copy()
    data['minutes'] = data['timestamp'].dt.hour * 60 + data['timestamp'].dt.minute

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(data['minutes'], y_data['building_baseline'], label='Baseline')
    ax.axhline(prediction, color='red')
    ax.set_title(f'{data.iloc[0]["date_site"]}')
    plt.show()


def show(data, time_col='timestamp', title=None, savings=None, save=False):

    fig, ax = plt.subplots(figsize=(14, 8))

    fields = ['building_baseline', 'offsets', 'target', 'soc',
              'threshold_old',
              'peak_prediction',
              'threshold', 'discharge_limits', 'charge_limits', 'temperature',]

    styles = {'threshold': {'color': 'brown',
                            'ls': '-',
                            'alpha': 1.0},
              'threshold_old': {'color': 'pink',
                                'ls': '-',
                                'alpha': 0.6},
              'peak_prediction': {'color': 'lightblue',
                                  'ls': '-',
                                  'alpha': 0.6}}


    if isinstance(data.index, pd.DatetimeIndex) and time_col not in data.columns:
        data[time_col] = data.index.to_series(keep_tz=True)


    for field in fields:
        alpha = 1.0
        if field == 'target' or field == 'peak_prediction' or field == 'threshold_old':
            alpha = 0.6
        ls = '-'
        if field == 'soc':
            ls = ':'
        if field in data.columns:
            data_y = np.array(data[field])
            if field == 'charge_limits':
                data_y = -data_y

            date_display = np.array([t.replace(tzinfo=None) for t in data[time_col]])

            data_with_period = [(y, p) for y, p in zip(data_y, data['period'])]
            if field == 'threshold' or field == 'threshold_old' or field == 'peak_prediction':

                # Create masks for thresholds so lines are discrete at different levels/periods
                for yv, p in list(set(data_with_period)):
                    if yv != np.nan:
                        # Create unique values mask
                        idx = data_y == yv
                        masks = []
                        start = None
                        end = None
                        for i, t in enumerate(idx):
                            if t and start is None:
                                start = i
                            if not t and start is not None and end is None:
                                end = i
                                mask = [True if ind >= start and ind < end else False for ind in range(len(idx))]
                                masks.append(np.array(mask))
                                start = None
                                end = None
                            if i == (len(idx) - 1) and t and start is not None:
                                end = i + 1
                                mask = [True if ind >= start and ind < end else False for ind in range(len(idx))]
                                masks.append(np.array(mask))

                        for mask in masks:
                            ax.plot(date_display[mask], data_y[mask], alpha=styles[field]['alpha'], ls=styles[field]['ls'],
                                    label=' '.join(field.split('_')).capitalize(), c=styles[field]['color'])

            #     ax.plot(data_y, alpha=alpha, ls=ls,
            #             label=' '.join(field.split('_')).capitalize())
            # else:
            else:

                ax.plot(date_display, data_y, alpha=alpha, ls=ls,
                        label=' '.join(field.split('_')).capitalize())

    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time")

    if title is None and 'date_site' in data.columns:
        title = f'{data.iloc[0]["date_site"]}'
    elif title is None:
        title = "Untitled"
    if savings:
        title = f'{title} - Savings: ${savings[0]:.2f} (D: ${savings[1]:.2f}, E: ${savings[2]:.2f})'
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    labels_set = []
    handles_set = []
    for label, handle in zip(labels, handles):
        if label in labels_set:
            continue
        labels_set.append(label)
        handles_set.append(handle)

    ax.legend(handles_set, labels_set, loc='best')
    ax.set_ylim(None, None)

    if save:
        plt.savefig(f"output/{title}.png")
    else:
        plt.show()


def lt_plot(site, data, t1=None, t2=None, max_peak=None, show=False):
    if t1 is None:
        t1 = data.index[0]
    if t2 is None:
        t2 = data.index[-1]

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.title(f"LT Optimization for {site}")

    # plotting power related graphs on first y axis
    ln1 = ax.plot(data.offsets[t1:t2],
                  label="Offset")
    try:
        ln2 = ax.plot(data.baseline[t1:t2],
                      label='Building Power')
    except AttributeError:
        ln2 = ax.plot(data.building_baseline[t1:t2],
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

    # start, end = ax.get_xlim()
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

    if show:
        plt.show()
    return ax


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error kW')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.ylim([0,50])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$kW^2$]')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.ylim([0,500])
  plt.legend()
  plt.show()