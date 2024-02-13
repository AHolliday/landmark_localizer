#!/usr/bin/env python

import argparse
import pickle
import os
import csv
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict

from sequence_experiment import SequenceExperimentResult, DummySeqExpResult
from landmark_localizer import transformations
from landmark_localizer import plot_utils as pu
from landmark_localizer import experimentUtils as expu
from landmark_localizer import geometryUtils as geo
from landmark_localizer import constants as cs


matplotlib.rc('font', size=15)


def cosine_error(pair_result):
    gt = None
    est = None
    if pair_result is not None:
        gt = pair_result['gt_T21']
        if 'est_T21' in pair_result:
            est = pair_result['est_T21']
    return geo.cosineDistance(gt, est)

    # if np.all(pair_result['gt_T21'] == 0) and \
    #     np.any( != 0):
    #     return 0.5
    # if pair_result is None or 'est_T21' not in pair_result:
    #     return 1
    # else:
    #     return distance.cosine(pair_result['gt_T21'].flatten(),
    #                            pair_result['est_T21'].flatten())


def unified_error(pair_result):
    # invert the transforms
    gt21 = geo.buildTfMatrix(pair_result['gt_R21'], pair_result['gt_T21'])
    est21 = geo.buildTfMatrix(pair_result['est_R21'], pair_result['est_T21'])
    return unified_error_from_transforms(gt21, est21)


def unified_error_from_transforms(gt_T, est_T):
    # Error = ||t_12g - t_12e||_2 + cosine_dist(q_12e, q_12g) * ||t_12e||_2
    t_err = np.linalg.norm(gt_T[:, 3] - est_T[:, 3])
    # scale them by the maximum of the ground truth and the estimated length.
    # rot_scale_factor = max(np.linalg.norm(est_T[:, 3]),
    #                        np.linalg.norm(gt_T[:, 3]))
    # actually, just scale by the ground-truth distance.
    rot_scale_factor = np.linalg.norm(gt_T[:, 3])
    quat_cos_dist = geo.rotationMatrixDistance(gt_T[:3, :3], est_T[:3, :3])
    return t_err + quat_cos_dist * rot_scale_factor


def pos_error(pair_result):
    return np.linalg.norm(pair_result['gt_T21'] - pair_result['est_T21'])


def relative_T_error(pair_result):
    return geo.relativeDistance(pair_result['gt_T21'], pair_result['est_T21'])


def rotational_error(pair_result):
    return geo.rotationMatrixDistance(pair_result['gt_R21'],
                                      pair_result['est_R21'])


def is_turn(pair_result, threshold=np.pi / 8):
    """Return whether the pair straddles a turned corner."""
    gt_R21 = pair_result['gt_R21']
    r, p, y = transformations.euler_from_matrix(gt_R21)
    return max(r, p, y) > threshold


def config_name(config, simple=False):
    if simple:
        return config['name']
    else:
        if config['extractor']['type'] in [expu.LIFT_STR, expu.SIFT_STR,
                                           expu.ORB_STR, expu.D2_STR]:
            return config['extractor']['type'].upper()
        else:
            try:
                # try to parse older configuration style
                net_path = config['extractor']['oeKwargs']['caffeProto']
                net_name = os.path.splitext(os.path.basename(net_path))[0]
            except KeyError:
                net_name = config['extractor']['type']
            try:
                # try to parse older configuration style
                blob_names = str(config['extractor']['oeKwargs']['featureBlobNames'])
            except KeyError:
                try:
                    blob_names = str(config['extractor']['oeKwargs']['features'])
                except KeyError:
                    blob_names = str(config['extractor']['oeKwargs']['feature'])

            oeKwargs = config['extractor']['oeKwargs']
            if 'patchSize' in oeKwargs:
                res_names = str(oeKwargs['patchSize'])
            else:
                res_names = str(oeKwargs)

            name_parts = [net_name, blob_names, res_names]
            name = ', '.join(name_parts)

            if config['extractor']['oeKwargs'].get('siftSubFeatures', False):
                name += ' with SIFT features'
            elif config['extractor']['oeKwargs'].get('d2SubFeatures', False):
                name += ' with D2 features'
            else:
                name += ' without sub-features'

            if config['localization']['subFeatureMetric'] == 'cosine':
                name += ' with cosine SIFT dist'
        return name


def is_failure(pair_result, threshold=None):
    if 'est_T21' not in pair_result:
        return True
    elif threshold and unified_error(pair_result) > threshold:
        return True
    return False


def plot_net_comparison(exp_results, err_function, err_name='Error',
                        failure_val=None, fig_num=1, save=False):
    # plt.figure(fig_num)
    net_error_df = pd.DataFrame(columns=['Net type', 'Layer', err_name])
    for exp_result in exp_results:
        values = dict()
        values[err_name] = get_mean_error(exp_result, err_function,
                                          failure_val)
        ext_cfg = exp_result.experiment_config['extractor']
        values['Net type'] = ext_cfg['type']
        values['Layer'] = ext_cfg['oeKwargs']['feature']
        net_error_df = net_error_df.append(values, ignore_index=True)

    # establish a "true" ordering over layers
    nettypes = net_error_df['Net type'].unique()
    netarchs = set(''.join(cc for cc in nt if not cc.isdigit())
                    for nt in nettypes)
    # make a separate comparison sub-plot for each net architecture family
    num_rows = int(np.ceil(np.sqrt(len(netarchs))))
    num_cols = int(np.ceil(len(netarchs) / num_rows))
    # fig, axes = plt.subplots(num_rows, num_cols)

    for ii, netarch in enumerate(netarchs):
        # ax = axes.flatten()[ii]
        _, ax = plt.subplots()
        netarch_df = net_error_df[net_error_df['Net type'].str.contains(netarch)]
        # remove any layers that don't appear at all in the experiments
        layer_order = [ll for ll in cs.LAYER_ORDERS[netarch]
                       if ll in netarch_df['Layer'].unique()]

        # plot a bar chart with the net types
        errors_by_label = []
        labels, groups = list(zip(*netarch_df.groupby('Net type')))

        # # TODO remove this line, it's just to help format VGG nicely
        # labels = [ll.upper() for ll in labels]

        for group in groups:
            # get a list of heights sorted by layer names
            errors = []
            for layer_name in layer_order:
                error_series = group.loc[group['Layer'] == layer_name][err_name]
                if len(error_series) == 0:
                    error = None
                else:
                    error = error_series.tolist()[0]
                errors.append(error)
            errors_by_label.append(errors)

        pu.bar_chart(errors_by_label, labels, layer_order, axis=ax)
        # TODO save if "save" is passed
        # plt.xlabel('Layer')
        # plt.ylabel(err_name)
        # ax.set_ylim(ymin=0, ymax=80)
        # plt.subplots_adjust(right=0.99, top=0.99)
        ax.legend(loc='lower right')
        # ax.set_title(netarch)
    # plt.tight_layout()


def plot_net_err_vs_size(exp_results, err_function, err_name='Error',
                         failure_val=None, fig_num=1, save=False):
    plt.figure(fig_num)
    net_error_df = pd.DataFrame(columns=['Net type', 'Layer', 'Size',
                                         err_name])
    # this is a hack...
    # blacklist = ['vgg11', 'vgg13', 'vgg16', 'resnet101', 'resnet152']
    blacklist = []
    # whitelist = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
    # whitelist = {'alexnet': ['pool2', 'conv3', 'conv4'],
    #              'resnet50': ['res3d, res4f'],
    #              'vgg16': ['pre_pool3', 'pool3', 'pre_pool4'],
    #              'densenet161': ['transition1', 'transition2', 'transition3'],
    #              'densenet169': ['transition1', 'transition2', 'transition3']}
    for exp_result in exp_results:
        values = dict()
        values[err_name] = get_mean_error(exp_result, err_function,
                                          failure_val)
        ext_cfg = exp_result.experiment_config['extractor']
        values['Net type'] = ext_cfg['type']
        # if values['Net type'] not in whitelist:
        #     continue
        values['Layer'] = ext_cfg['oeKwargs']['feature']
        values['Size'] = cs.NET_LAYER_SIZES[values['Net type']][values['Layer']]
        num_failures = len([rr for _, rr in exp_result.pair_results
                            if is_failure(rr)])
        if num_failures > 0 or values['Net type'] in blacklist:
            continue
        # if values['Net type'] in whitelist and \
        #    values['Layer'] in whitelist[values['Net type']]:
        net_error_df = net_error_df.append(values, ignore_index=True)

    point_styles = ["o", "v", "X", "s", "d", "^", "x", "+", "*", "P"]
    # nettypes = net_error_df['Net type'].unique()
    # netarchs = list(set(''.join(cc for cc in nt if not cc.isdigit())
    #                     for nt in nettypes))
    netarchs = ['alexnet', 'resnet', 'vgg', 'densenet']
    netarch_counts = defaultdict(int)
    for name, group in net_error_df.groupby('Net type'):
        netarch = ''.join(cc for cc in name if not cc.isdigit())
        colour_idx = netarchs.index(netarch)
        # each colour is one architecture family, each style one architecture
        colour = 'C{}'.format(colour_idx)
        style = point_styles[netarch_counts[netarch]]
        netarch_counts[netarch] += 1
        plt.scatter(group['Size'], group[err_name], label=name, marker=style,
                    color=colour)
        # for _, row in group.iterrows():
        #     loc = (row['Size'], row[err_name])
        #     plt.annotate(row['Layer'], loc)

    # # plot the convex hull
    # points = np.zeros((len(net_error_df), 2))
    # points[:, 0] = net_error_df['Size']
    # points[:, 1] = net_error_df[err_name]
    # hull = ConvexHull(points)
    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    ax = plt.subplot(111)
    chart_box = ax.get_position()
    ax.set_position([chart_box.x0, chart_box.y0, chart_box.width*0.6,
                     chart_box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8))
    # plt.legend(loc='best')
    # plt.yscale('log')
    # plt.xlim(left=0, right=None)
    # plt.ylim(bottom=45, top=60)
    plt.xlabel('Feature size (# floating point values)')
    plt.ylabel(err_name)
    plt.tight_layout()



def plot_error_by_gt_dist(exp_results, err_function, failure_val=1,
                          fig_num=1, save=False):
    """Useful for finding which examples are anomalous.

    Makes a scatter plot of all pairs, and you can mouse over each point to
    see what pair it corresponds to.
    """
    plt.figure(fig_num)
    fig, ax = plt.subplots()
    annot = ax.annotate("", xy=(0,0), xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    for exp_result in exp_results:
        gt_dists = []
        errors = []
        pairs = []
        for pair, pair_result in exp_result.pair_results:
            gt_dists.append(np.linalg.norm(pair_result['gt_T21']))
            pairs.append(tuple(map(str, pair)))
            if is_failure(pair_result):
                errors.append(failure_val)
            else:
                errors.append(err_function(pair_result))

        label = config_name(exp_result.experiment_config, SIMPLE_NAME)
        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}, {}".format(" ".join([pairs[n][0] for n in ind["ind"]]),
            " ".join([pairs[n][1] for n in ind["ind"]]))
            annot.set_text(text)
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            # annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        sc = plt.scatter(gt_dists, errors, label=label, marker='.')
        fig.canvas.mpl_connect("motion_notify_event", hover)

        # fitXs, fitYs = pu.get_plottable_fit_curve(gt_dists, errors,
        #                                           dim=1)
        # plt.plot(fitXs, fitYs, label=label)

    title = exp_results[0].data_name + ' ' + err_function.__name__
    plt.title(title)
    plt.legend()
    plt.show()
    if save:
        plt.savefig(title.replace(' ', '_') + '.png')


def plot_smoothed_error(exp_results, err_function, sigma=20, failure_val=1,
                        fig_num=1, save=False, yname=None, log_scale=False):
    plt.figure(fig_num)
    for exp_result in exp_results:
        pair_results = [rr for _, rr in exp_result.pair_results]
        pair_results = sorted(pair_results,
                              key=lambda rr: np.linalg.norm(rr['gt_T21']))
        sep_dists = []
        errors = []
        for rr in pair_results:
            if is_failure(rr):
                if failure_val is not None:
                    errors.append(failure_val)
                    sep_dists.append(np.linalg.norm(rr['gt_T21']))
            else:
                errors.append(err_function(rr))
                sep_dists.append(np.linalg.norm(rr['gt_T21']))

        smoothed_errors = gaussian_filter1d(errors, sigma=sigma)

        label = config_name(exp_result.experiment_config, SIMPLE_NAME)
        plt.plot(sep_dists, smoothed_errors, label=label)

    plt.legend()
    if log_scale:
        plt.yscale('log')

    if yname:
        plt.ylabel(yname)
    else:
        plt.ylabel(err_function.__name__)
    plt.xlabel('Distance between images in pairs (m)')
    if save:
        plt.savefig(title.replace(' ', '_') + '.png')
    plt.tight_layout()


def plot_error_by_gap(exp_results, err_function, fit_log, failure_val=1,
                      fig_num=1, save=False, yname=None,
                      log_scale=False, use_violin=False):
    plt.figure(fig_num)
    def get_assymetric_bars(values):
        avg = np.median(values)
        below = np.mean([vv for vv in values if vv < avg])
        above = np.mean([vv for vv in values if vv > avg])
        return avg - below, above - avg

    labels = []
    all_gap_errors = []
    # colors = ['C4', 'C2', 'C5']
    for ii, exp_result in enumerate(exp_results):
        gap_results = group_by_dist(exp_result, num_bins=8)
        gap_errors = []
        dists = sorted(gap_results.keys())

        # compute errors
        for jj, dist in enumerate(dists):
            errors = []
            for rr in gap_results[dist]:
                if is_failure(rr):
                    if failure_val is not None:
                        errors.append(failure_val)
                else:
                    errors.append(err_function(rr))
            gap_errors.append(errors)
        all_gap_errors.append(gap_errors)

        label = config_name(exp_result.experiment_config, SIMPLE_NAME)
        labels.append(label)
        if not use_violin:
            bars = np.array([get_assymetric_bars(ge) for ge in gap_errors]).T
            # bars = np.array([np.std(ge) for ge in gap_errors])
            ys = [np.mean(ge) for ge in gap_errors]
            plt.errorbar(dists, ys, bars, label=label, fmt='--o', capsize=5)
                         # color=colors[ii])

    if use_violin:
        pu.violin_plot(all_gap_errors, labels, dists, log_scale=log_scale,
                       legend_loc='upper left')
    else:
        plt.legend()
        if log_scale:
            plt.yscale('log')
    title = exp_results[0].data_name + ' ' + err_function.__name__
    if yname:
        plt.ylabel(yname)
    else:
        plt.ylabel(err_function.__name__)
    plt.xlabel('Distance between images in pairs (m)')
    if save:
        plt.savefig(title.replace(' ', '_') + '.png')
    plt.tight_layout()


def get_mean_error(exp_result, err_function, failure_val=1):
    errors = [err_function(r) if not is_failure(r) else failure_val
              for _, r in exp_result.pair_results]
    errors = [ee for ee in errors if ee is not None]
    return np.mean(errors)


def print_errors(exp_results, err_function, failure_val=1):
    names_and_errors = []
    for exp_result in exp_results:
        mean_error = get_mean_error(exp_result, err_function, failure_val)
        cfg_name = config_name(exp_result.experiment_config, SIMPLE_NAME)
        names_and_errors.append((cfg_name, mean_error))

    print('In increasing order:')
    for cfg_name, err in sorted(names_and_errors, key=lambda x: x[1]):
        print(cfg_name, 'average score is {:03f}'.format(err), 'on', \
            err_function.__name__)


def plot_failure_counts_by_gap(exp_results, threshold=None, fig_num=1,
                               save=False, simple=False):
    heights_by_label = []
    labels = []
    for ii, exp_result in enumerate(exp_results):
        gap_results = group_by_dist(exp_result, num_bins=8)
        failure_rates = []
        for gap, results in gap_results.items():
            num_failures = len([r for r in results
                                if is_failure(r, threshold)])
            failure_rates.append(num_failures)
        heights_by_label.append(failure_rates)
        labels.append(config_name(exp_result.experiment_config, simple))

    gaps = sorted(gap_results.keys())
    x_tick_labels = list(map('{:0.1f}'.format, gaps))
    pu.bar_chart(heights_by_label, labels, x_tick_labels)

    title = exp_results[0].data_name + " failure rates"
    if not simple:
        plt.title(title)
    plt.ylabel('# localization failures')
    plt.xlabel('Distance between images in pairs (m)')
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig(title.replace(' ', '_') + '.png')


def print_failure_counts(exp_results, threshold=None):
    names_and_counts = []
    for exp_result in exp_results:
        num_failures = len([r for _, r in exp_result.pair_results
                            if is_failure(r, threshold)])
        cfg_name = config_name(exp_result.experiment_config, SIMPLE_NAME)
        names_and_counts.append((cfg_name, num_failures))
    for cfg_name, count in sorted(names_and_counts, key=lambda x: x[1]):
        print(cfg_name, 'has', count, 'failures')


def group_by_gap(exp_result):
    gap_results = {}
    for pair, result in exp_result.pair_results:
        gap = pair[1] - pair[0]
        if gap not in gap_results:
            gap_results[gap] = []
        gap_results[gap].append(result)
    return gap_results


def group_by_dist(exp_result, num_bins=10):
    """Group the results by the true distances between the pairs, into bins
    of equal size."""
    results = [rr for _, rr in exp_result.pair_results]
    results = sorted(results, key=lambda xx: np.linalg.norm(xx['gt_T21']))
    bin_size = len(results) // num_bins
    gap_dict = {}
    for chunk in range(num_bins):
        chunk_results = results[chunk * bin_size:(chunk + 1) * bin_size]
        chunk_med = np.median([np.linalg.norm(rr['gt_T21'])
                               for rr in chunk_results])
        gap_dict[chunk_med] = chunk_results
    return gap_dict


def plot_dists_by_gap(exp_results, fig_num=1):
    plt.figure(fig_num)
    col_width = DEFAULT_COL_WIDTH
    exp_result = exp_results[0]

    # group results by gap
    gap_results = group_by_dist(exp_result)

    # compute failure rate in each gap
    gaps = sorted(gap_results.keys())
    x_pos = np.arange(len(gaps))
    mean_gap_dists = []
    std_gap_dists = []
    for gap, results in gap_results:
        gap_dists = [np.linalg.norm(r['gt_T21']) for r in results]
        mean_gap_dists.append(np.mean(gap_dists))
        std_gap_dists.append(np.std(gap_dists))

    plt.bar(x_pos, mean_gap_dists, col_width, label='average gap dists',
            tick_label=(list(map(str, gaps))), yerr=std_gap_dists)
    plt.title(exp_result.data_name)
    plt.legend()


def merge_results_from_same_config(exp_results):
    config_results = {}
    cfg_names_to_configs = {}
    for exp_result in exp_results:
        cfg_name = config_name(exp_result.experiment_config, SIMPLE_NAME)
        if cfg_name not in config_results:
            config_results[cfg_name] = []
            cfg_names_to_configs[cfg_name] = exp_result.experiment_config
        config_results[cfg_name] += exp_result.pair_results

    merged_exp_results = []
    for cfg_name, pair_results in config_results.items():
        cfg = cfg_names_to_configs[cfg_name]
        merged_exp_results.append(DummySeqExpResult('All', cfg, pair_results))
    return merged_exp_results


def find_longest_transforms(exp_result):
    max_pr = max(exp_result.pair_results,
                 key=lambda x: np.linalg.norm(x[1]['gt_T21']))
    max_pair, max_result = max_pr
    print('max dist:', np.linalg.norm(max_result['gt_T21']), 'over gap', \
        max_pair[1] - max_pair[0])


def results_to_csv(exp_results, err_function_dict, out_filename):
    def flatten_config(cfg, prefix=''):
        # convert a nested configuration dict into a flat dict
        flat_cfg = {}
        for key, value in cfg.items():
            if type(value) is dict:
                # recurse and add results to this flat config
                sub_prefix = prefix + str(key) + '__'
                sub_cfg = flatten_config(value, prefix=sub_prefix)
                for sk, sv in sub_cfg.items():
                    flat_cfg[sk] = sv
            else:
                flat_cfg[prefix + str(key)] = value
        return flat_cfg

    flat_config_keys = flatten_config(exp_results[0].experiment_config).keys()
    err_keys = err_function_dict.keys()
    fieldnames = list(flat_config_keys) + list(err_keys) + ['num failures']
    with open(out_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for exp_result in exp_results:
            # flatten the config to a flat dictionary
            row_dict = flatten_config(exp_result.experiment_config)
            # compute errors and add them to the csv
            for err_key in err_keys:
                error = get_mean_error(exp_result, err_function_dict[err_key])
                row_dict[err_key] = error
            # compute number of failures and add to the csv
            num_failures = len([rr for _, rr in exp_result.pair_results
                                if is_failure(rr)])
            row_dict['num failures'] = num_failures
            writer.writerow(row_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', action='append', help='\
Pickle files containing the results to plot.')
    parser.add_argument('--keepturns', action='store_true', help='\
If provided, do not ignore transforms that round a corner.')
    parser.add_argument('--merge', action='store_true', help='\
Combine results from experiments with identical configs.')
    parser.add_argument('-f', '--fit', action='store_true', help='\
If provided, fit a curve to the mean errors at each gap distance.')
    parser.add_argument('-t', '--threshold', type=float, help='\
If provided, consider points with error above this to be failures.')
    parser.add_argument('-s', '--save', action='store_true', help='\
If provided, save figures as .pngs.')
    parser.add_argument('--simple', action='store_true', help='\
If provided, use simple plot labels.')
    parser.add_argument('--rename', action='store_true', help='\
If provided, prompt the user for new display names for each config.')
    parser.add_argument('--csv', help='\
Path to a csv file in which to save the results.')
    parser.add_argument('--nets', action='store_true', help='\
If provided, plot a network layer comparison.')
    args = parser.parse_args()
    all_exp_results = []
    for pkl_filename in args.r:
        with open(pkl_filename, 'rb') as f:
            all_exp_results += pickle.load(f)

    global SIMPLE_NAME
    SIMPLE_NAME = args.simple

    if args.merge:
        all_exp_results = merge_results_from_same_config(all_exp_results)

    if not args.keepturns:
        # filter out turns
        for exp_result in all_exp_results:
            results = exp_result.pair_results
            no_turns = [(p, r) for p, r in results if not is_turn(r)]
            exp_result.pair_results = no_turns

    print(len(all_exp_results[0].pair_results), "pairs were considered.")
    print(sum([len(r.pair_results) for r in all_exp_results]), \
         "transforms were estimated.")

    if args.rename and args.simple:
        # renaming doesn't do anything if not using simple names
        for exp_result in all_exp_results:
            cfg = exp_result.experiment_config
            full_name = config_name(cfg)
            new_name = input('New name for config ' + full_name + ': ')
            if new_name:
                cfg['name'] = new_name
            else:
                cfg['name'] = full_name
            print(full_name, 'now called', cfg['name'])

    fig_sequence = iter(list(range(10000)))
    # plot_dists_by_gap(all_exp_results, next(fig_sequence))

    if args.nets:
        plot_net_err_vs_size(all_exp_results, unified_error,
                            'Mean pose error (m)',
                             fig_num=next(fig_sequence), save=args.save)
        # plot_net_err_vs_size(all_exp_results, rotational_error,
        #                     'Rotational error',
        #                      fig_num=next(fig_sequence), save=args.save)
        plot_net_comparison(all_exp_results, unified_error,
                            'Mean pose error (m)',
                             fig_num=next(fig_sequence), save=args.save)
        # plot_net_comparison(all_exp_results, rotational_error,
        #                     'Rotational error', fig_num=next(fig_sequence),
        #                     save=args.save)
        failure_count = lambda x: 0
        plot_net_comparison(all_exp_results, failure_count, 'Failure rate',
                            failure_val=1, fig_num=next(fig_sequence),
                            save=args.save)

    else:
        plot_error_by_gap(all_exp_results, unified_error, args.fit,
                          fig_num=next(fig_sequence), save=args.save,
                          failure_val=None, yname='pose error (m)',
                          log_scale=True)
        plot_smoothed_error(all_exp_results, unified_error, failure_val=None,
                            yname='pose error (m)', log_scale=False)

        # plot_error_by_gap(all_exp_results, rotational_error, args.fit,
        #                   fig_num=next(fig_sequence), save=args.save,
        #                   simple=args.simple, yname='Rotational error')
        plot_failure_counts_by_gap(all_exp_results, args.threshold,
                                   fig_num=next(fig_sequence), save=args.save,
                                   simple=args.simple)

        if args.csv:
            error_dict = {'relative T error': relative_T_error,
            'rotational error': rotational_error,
            'combined error': unified_error}
            results_to_csv(all_exp_results, error_dict, args.csv)

    print_errors(all_exp_results, unified_error)
    print_errors(all_exp_results, rotational_error)
    print_failure_counts(all_exp_results, args.threshold)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
