import json
from datetime import datetime

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys 
import pandas as pd



def get_csvs(directory, fname="progress.csv"):
    return [os.path.join(x[0], fname) for x in os.walk(directory) if fname in x[2]]

def get_dataframes(path, filtered=False, **kwargs):
    dfs = {}
    for i, csv in enumerate(get_csvs(path, **kwargs)):
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
    b = customized_box_plot(lengths.values[5::5], ax, redraw=True, notch=0, vert=1, whis=1.5)
    # ax.set_ylim([0, ymax])
    _xticks = np.hstack([[0], (range(5, len(lengths.values), 5))])
    plt.xticks(range(len(_xticks)), _xticks, size='x-small')


#### PLOT TIMING

convert = lambda timestr: datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
JSON_CACHE = {}

def convert_relative_time(dict_times):
    assert "total" in dict_times
    converted = {}
    start, end = (convert(x) for x in dict_times['total'])
    rel_time = lambda ts: (convert(ts) - start).total_seconds()
    for i, k in enumerate(sorted(dict_times)):
        valarr = dict_times[k]
        if k == "total":
            # plt.plot(rel_time(valarr[1]), i + 1, 'o')
            continue

        converted[str(k)] = np.array([[0, rel_time(t0), rel_time(t1)] for t0, t1 in valarr])
    
    for i, k in enumerate(sorted(converted, reverse=True, key=lambda y: np.max(converted[y][:, 2]))):
        converted[k][:, 0] = i + 1

    return converted

def efficiency(converted_dict, num_drop=0):
    worked, total_time = worked_times(converted_dict, num_drop)
    return worked / total_time #, _nplastruns

def worked_times(converted_dict, num_drop=0):
    """Returns real amount of time worked (sum of real time), total time taken * num cores"""
    assert 0 <= num_drop < len(converted_dict), "Can't drop that many" 
    last_runs = [np.max(v, axis=0)[1:] for k, v in converted_dict.items()] #array of (start, end) of last runs for each worker
    last_runs.sort(reverse=True, key=lambda r: r[1]) # sort by ending time
    _nplastruns = np.asarray(last_runs) # converts to numpy array for fast indexing
    worked = sum(_nplastruns[:num_drop, 0]) + sum(_nplastruns[num_drop:, 1]) #truncate ones that are not done, keep ones that are
    total_time = _nplastruns[num_drop, 1] * len(converted_dict) 
    return worked, total_time

avg = lambda x, N: np.convolve(x, np.ones((N,))/N, mode='valid')

def get_reltimedict_from_path(exp_path):
    if exp_path not in JSON_CACHE:
        with open(exp_path + "times_0.json") as f:
            times_w0 = json.load(f)
        with open(exp_path + "times_1.json") as f:
            times_w1 = json.load(f)
        JSON_CACHE[exp_path] = [times_w0, times_w1]
    else:
        times_w0, times_w1 = JSON_CACHE[exp_path]

    converted_times = [convert_relative_time(tdict) for tdict in times_w0['timing']]
    converted_times.extend([convert_relative_time(tdict) for tdict in times_w1['timing']])
    return converted_times

def efficiency_matrix(exp_path):
    converted_times = get_reltimedict_from_path(exp_path)
    NUM_CORES = len(converted_times[0])

    eff_matrix = np.asarray([[efficiency(cdict, core) for core in range(NUM_CORES)] 
                                                    for cdict in converted_times])
    return eff_matrix

def max_efficiency_plot(exp_path):
    df = dataframe_average(exp_path)
    eff_matrix = efficiency_matrix(exp_path)
    NUM_CORES = eff_matrix.shape[1]
    plt.plot(np.argmax(eff_matrix, axis=1), df.AvgTrajLen, 'o', alpha=0.3)
    plt.xlim([-0.5, NUM_CORES])


def min_efficiency_plot(exp_path):
    df = dataframe_average(exp_path)
    eff_matrix = efficiency_matrix(exp_path)
    NUM_CORES = eff_matrix.shape[1]
    plt.plot(np.argmin(eff_matrix, axis=1), df.AvgTrajLen, 'o', alpha=0.3)
    plt.xlim([-0.5, NUM_CORES])

def plot_timechart(orig_dict, idx):

    def plot_converted_times(converted_times):
        values = np.vstack([v for v in converted_times.values()])
        worker, start, end = values[:, 0], values[:, 1], values[:, 2]
        plt.hlines(worker, start, end)
        plt.plot(start, worker, 'b^')

    plt.figure(figsize=[15, 10])

    plt_times = convert_relative_time(orig_dict['timing'][idx])
    plot_converted_times(plt_times)
    plt.ylim([0, len(orig_dict['timing'][idx])])

def plot_timechart_file(pth, idx):
    with open(pth) as f:
        times_w1 = json.load(f)

    plot_timechart(times_w1, idx)

