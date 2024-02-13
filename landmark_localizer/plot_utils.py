import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def get_plottable_fit_curve(xs, ys, dim, nSamples=200):
    params = np.polyfit(xs, ys, dim)
    xSamples = np.linspace(min(xs), max(xs), nSamples)
    curveSamples = [sum([p * (x**i) for i, p in enumerate(reversed(params))])
                    for x in xSamples]
    return xSamples, curveSamples


def box_plot(dataframes, legend_labels, xkey, ykey, log_scale=False):
    df = pd.concat(dataframes, keys=legend_labels).reset_index()

    if log_scale:
        df[ykey] = np.log10(df[ykey])
        min_tick = np.floor(df[ykey].min())
        max_tick = np.floor(df[ykey].max())
        ytick_locs = np.arange(min_tick, max_tick + 1)
        ytick_labels = ["$10^{" + str(int(yt)) + "}$" for yt in ytick_locs]
        plt.yticks(ytick_locs, ytick_labels)

    sns.boxplot(x=xkey, y=ykey, hue='level_0', data=df)
    plt.legend()


def violin_plot(dataset_sets, legend_labels, dists=None, ds_labels=None,
                log_scale=False, legend_loc=None, add_legend=True):
    if log_scale:
        # display powers-of-10 as the xticks
        min_val = min(min(min(ds) for ds in dss) for dss in dataset_sets)
        min_tick = np.floor(np.log10(min_val))
        max_val = max(max(max(ds) for ds in dss) for dss in dataset_sets)
        max_tick = np.ceil(np.log10(max_val))
        ytick_locs = np.arange(min_tick, max_tick + 1)
        ytick_labels = ["$10^{" + str(int(yt)) + "}$" for yt in ytick_locs]
        plt.yticks(ytick_locs, ytick_labels)
        # log-scale the data
        dataset_sets = [[np.log10(ds) for ds in datasets]
                        for datasets in dataset_sets]

    # determine the base width of the violins
    base_width = 1
    if dists is not None:
        base_width = np.diff(dists).min()

    width = 0.5
    if dists is not None:
        width = np.diff(dists).min()

    if dists is None:
        base_positions = np.arange(1, max([len(ds) for ds in dataset_sets]) + 1)
    else:
        base_positions = np.array(dists)

    offsets = []
    num_offsets = 1 + len(dataset_sets) // 2
    offset_size = 1 / num_offsets

    for ii, _ in enumerate(dataset_sets):
        offsets.append((1 + (ii // 2)) * offset_size)

    # plot the violins
    handles = []
    for offset, legend_label, datasets in zip(offsets, legend_labels,
                                               dataset_sets):
        # sometimes underflows can occur in violinplot.  That's fine.
        positions = base_positions - 0.5 + offset
        with np.errstate(under='warn'):
            vp = plt.violinplot(datasets, positions=positions, widths=width*0.5,
                                showmeans=False)
        colour = vp['bodies'][0].get_facecolor()[0]

        ds_means = [np.mean(ds) for ds in datasets]
        plt.scatter(positions, ds_means, s=40, color=colour, alpha=1)

        handles.append(mpatches.Patch(color=colour, label=legend_label))

    if dists is None or ds_labels is not None:
        # set the x-ticks
        if ds_labels is None:
            ds_labels = [str(dd) for dd in dists]
        # plt.xticks(plt.xticks()[0][1:], labels=ds_labels)
        plt.xticks(base_positions, labels=ds_labels)

    if add_legend:
        plt.legend(handles=handles, loc=legend_loc)


def vertical_scatters_plot(dataset_sets, legend_labels, dists=None,
                           ds_labels=None, log_scale=False, legend_loc=None,
                           add_legend=True):
    if log_scale:
        # display powers-of-10 as the xticks
        min_val = min(min(min(ds) for ds in dss) for dss in dataset_sets)
        min_tick = np.floor(np.log10(min_val))
        max_val = max(max(max(ds) for ds in dss) for dss in dataset_sets)
        max_tick = np.ceil(np.log10(max_val))
        ytick_locs = np.arange(min_tick, max_tick + 1)
        ytick_labels = ["$10^{" + str(int(yt)) + "}$" for yt in ytick_locs]
        plt.yticks(ytick_locs, ytick_labels)
        # log-scale the data
        dataset_sets = [[np.log10(ds) for ds in datasets]
                        for datasets in dataset_sets]

    # determine the base width of the violins
    base_width = 1
    if dists is not None:
        base_width = np.diff(dists).min()

    width = 0.5
    if dists is not None:
        width = np.diff(dists).min()

    # plot the violins
    handles = []
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colours = colour_cycle[:len(dataset_sets)]
    width = 0.5
    for dsi, datasets in enumerate(dataset_sets):
        legend_label = legend_labels[dsi]
        colour = colours[dsi]
        offset = width * (dsi + 0.5) / len(dataset_sets)

        # sometimes underflows can occur in violinplot.  That's fine.
        with np.errstate(under='warn'):

            for ii, ds in enumerate(datasets):
                xs = [ii + offset] * len(ds)
                vp = plt.scatter(xs, ds, alpha=0.1, color=colour)
        handles.append(mpatches.Patch(color=colour, label=legend_label))

    if dists is None or ds_labels is not None:
        # set the x-ticks
        if ds_labels is None:
            ds_labels = [str(dd) for dd in dists]
        plt.xticks(plt.xticks()[0][1:], labels=ds_labels)


    if add_legend:
        plt.legend(handles=handles, loc=legend_loc)


def bar_chart(heights_by_label, labels, x_tick_labels, col_width=0.7,
              axis=None):
    if axis is None:
        _, axis = plt.subplots()
    bar_width = col_width / len(labels)
    x_pos = np.arange(len(x_tick_labels))

    for ii, (label, heights) in enumerate(zip(labels, heights_by_label)):
        # filter out elements where height is None, as this indicates
        # no data
        poses = [pp for ii, pp in enumerate(x_pos + ii * bar_width)
                 if heights[ii] is not None]
        these_xticks = [ll for ii, ll in enumerate(x_tick_labels)
                        if heights[ii] is not None]
        heights = [hh for hh in heights if hh is not None]
        axis.bar(poses, heights, bar_width, label=label)
    axis.set_xticks(x_pos + ii * bar_width / 2)
    axis.set_xticklabels(x_tick_labels, rotation=20)
    # import pdb; pdb.set_trace()
