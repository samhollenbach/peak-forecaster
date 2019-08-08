import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import pandas as pd

def baseline_plot(x_data, y_data, prediction):
    data = x_data.copy()
    data['minutes'] = data['timestamp'].dt.hour * 60 + data['timestamp'].dt.minute

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(data['minutes'], y_data['building_baseline'], label='Baseline')
    ax.axhline(prediction, color='red')
    ax.set_title(f'{data.iloc[0]["date_site"]}')
    plt.show()


def baseline_plot2(data, title=None, savings=None, save=False):

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(data['timestamp'], data['building_baseline'], label='Baseline')
    ax.plot(data['timestamp'], data['offsets'], label='Offset')
    ax.plot(data['timestamp'], data['target'], label='Target', alpha=0.6)
    ax.plot(data['timestamp'], data['soc'], '--', label='SOC')
    if 'threshold_old' in data.columns:
        ax.plot(data['timestamp'], data['threshold_old'], label='Threshold Old')
    if 'threshold' in data.columns:
        ax.plot(data['timestamp'], data['threshold'], label='Threshold New')
    ax.plot(data['timestamp'], data['discharge_limits'], label='Discharge Limit')
    ax.plot(data['timestamp'], -data['charge_limits'], label='Charge Limit')
    ax.plot(data['timestamp'], data['temperature'], label='Temperature')
    if 'peak_prediction' in data.columns:
        ax.plot(data['timestamp'], data['peak_prediction'], label='Predicted Peaks')

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
    ax.legend(loc='lower right')
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