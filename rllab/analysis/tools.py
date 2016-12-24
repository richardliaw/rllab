import json
from datetime import datetime

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys 
import pandas as pd



def get_csvs(directory):
    return [os.path.join(x[0], "progress.csv") for x in os.walk(directory) if "progress.csv" in x[2]]

def get_dataframes(path, filtered=False):
    dfs = {}
    for i, csv in enumerate(get_csvs(path)):
        print csv
        df_csv = pd.read_csv(csv)
        if filtered:
            if df_csv['AverageReturn'][77] < -50:
                print 'True'
                continue
        dfs[csv] =  df_csv
    return dfs

def dataframe_average(path, filtered=False):
    dfs = get_dataframes(path, filtered=filtered)
    panel=pd.Panel(dfs)
    return panel.mean(axis=0)

def wasted_work(v):
    return (v['SampleTimeTaken'] - v['BatchLimitTime']) /v['SampleTimeTaken']

def plot_throughput(v, label=None):
    plt.plot(np.cumsum(v['SampleTimeTaken']), np.cumsum(v['TotalSamples']), label=label)


def customized_box_plot(percentiles, axes, redraw = True, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """
    n_box = len(percentiles)
    box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs) 
    # Creates len(percentiles) no of box plots

    min_y, max_y = float('inf'), -float('inf')
    fliers_xy = None

    for box_no, (q1_start, 
                 q2_start,
                 q3_start,
                 q4_start,
                 q4_end,
                 # fliers_xy
                ) in enumerate(percentiles):
        # Lower cap
        box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])

        # Higher cap
        box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])

        # Box
        box_plot['boxes'][box_no].set_ydata([q2_start, 
                                             q2_start, 
                                             q4_start,
                                             q4_start,
                                             q2_start])

        # Median
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

        # Outliers
        if fliers_xy is not None and len(fliers_xy[0]) != 0:
            # If outliers exist
            box_plot['fliers'][box_no].set(xdata = fliers_xy[0],
                                           ydata = fliers_xy[1])

            min_y = min(q1_start, min_y, fliers_xy[1].min())
            max_y = max(q4_end, max_y, fliers_xy[1].max())

        else:
            min_y = min(q1_start, min_y)
            max_y = max(q4_end, max_y)

        # The y axis is rescaled to fit the new box plot completely with 10% 
        # of the maximum value at both ends
        axes.set_ylim([min_y*1.1, max_y*1.1])

    # If redraw is set to true, the canvas is updated.
    if redraw:
        axes.figure.canvas.draw()

    return box_plot

def plot_trajectory_distributions(df, xmax=80, ymax=1200):
    lengths = df[['MinTrajLen', 'Q1TrajLen', 'AvgTrajLen', 'Q3TrajLen', 'MaxTrajLen']]
    fig, ax = plt.subplots(figsize=(10, 5))
    b = customized_box_plot(lengths.values[1::5], ax, redraw=True, notch=0, vert=1, whis=1.5)
    ax.set_ylim([0, ymax])
    _xticks = np.hstack([[0], (range(1, xmax, 5))])
    plt.xticks(range(len(_xticks)), _xticks, size='x-small')


#### PLOT TIMING

convert = lambda timestr: datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")

def plot_timechart(orig_dict, idx):
    def convert_relative_time(dict_times):
        assert "total" in dict_times
        converted = {}
        start, end = (convert(x) for x in dict_times['total'])
        rel_time = lambda ts: (convert(ts) - start).total_seconds()
        for i, (k, valarr) in enumerate(dict_times.items()):
            if k == "total":
                continue
            converted[str(k)] = np.array([[i + 1, rel_time(t0), rel_time(t1)] for t0, t1 in valarr])
        return converted

    def plot_converted_times(converted_times):
        values = np.vstack([v for v in plt_times.values()])
        worker, start, end = values[:, 0], values[:, 1], values[:, 2]
        plt.hlines(worker, start, end)
        plt.plot(start, worker, 'b^')

    plt.figure(figsize=[20, 10])

    plt_times = convert_relative_time(orig_dict['timing'][idx])
    plot_converted_times(plt_times)
    plt.ylim([0, len(orig_dict['timing'][idx]) + 1])

def plot_timechart_file(pth, idx):
    with open(pth) as f:
        times_w1 = json.load(f)

    plot_timechart(times_w1, idx)

