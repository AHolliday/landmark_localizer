import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

from landmark_localizer import constants as cs
from landmark_localizer import plot_utils as plu
from sequence_experiment import ColdDatasetSequence, KittiSequence
from kitti_pr_experiment import MultiDistPrResultsContainer, \
    FabmapResultsContainer, PrLocResult
from plot_sequence_results import unified_error_from_transforms

matplotlib.rc('font', size=15)


def get_multi_dist_results_df(mdprc, step_exclude_list=[], sim_threshold=None):
    data_seq = mdprc.data_seq
    def query_result_to_dict(step_dist_m, dist_result, index):
        qi, ci = dist_result.matches[index]
        sub_idxs = dist_result.sub_idxs
        try:
            idx_dist = abs(qi - sub_idxs[index + 1])
        except IndexError:
            idx_dist = abs(qi - sub_idxs[index - 1])
        data_dict = {
            cs.STEP_KEY: step_dist_m,
            cs.NEAR_I_DIST_KEY: idx_dist,
            cs.NEW_PROB_KEY: 0,
        }
        if type(data_seq) is ColdDatasetSequence:
            data_dict[cs.QROOM_KEY] = data_seq.get_room_id(qi)
            data_dict[cs.QROOM_TYPE_KEY] = data_seq.get_room_type(qi)

        if ci is None:
            # no consistent match was found, so call this a new scene
            data_dict[cs.NEW_PROB_KEY] = 1
            print('failure at', data_seq.get_image_paths()[qi])
        else:
            est_T_21 = dist_result.est_poses[index]
            gt_T_w1 = data_seq.get_true_pose(qi)
            gt_T_2w = np.linalg.inv(data_seq.get_true_pose(ci))
            gt_T_21 = gt_T_2w.dot(gt_T_w1)
            gtp = gt_T_21[:3, 3]
            gtr = gt_T_21[:3, :3]
            data_dict[cs.DIST_KEY] = np.linalg.norm(gtp)
            if est_T_21 is not None:
                etp = est_T_21[:3, 3]
                etr = est_T_21[:3, :3]
                data_dict[cs.EST_DIST_KEY] = np.linalg.norm(etp)
                data_dict[cs.SYM_ERR_KEY] = \
                    unified_error_from_transforms(gt_T_21, est_T_21)
                # data_dict[cs.ABS_POS_ERR_KEY] = np.linalg.norm(gtp - etp)
                # data_dict[cs.REL_POS_ERR_KEY] = geo.relativeDistance(gtp, etp)
                # data_dict[cs.ROT_ERR_KEY] = geo.rotationMatrixDistance(gtr, etr)
                # data_dict[cs.COS_ERR_KEY] = geo.cosineDistance(gtp, etp)
            else:
                # no consistent transform could be found with the match
                print('failure at', data_seq.get_image_paths()[qi],
                      data_seq.get_image_paths()[ci])
                data_dict[cs.NEW_PROB_KEY] = 1

            # record the simulation score
            if type(mdprc) is FabmapResultsContainer:
                sim_score = dist_result.conf_matrix[qi, ci]
            else:
                sim_score = mdprc.sim_matrix[qi, ci]
                if sim_threshold and sim_score <= sim_threshold:
                    data_dict[cs.NEW_PROB_KEY] = 1

            data_dict[cs.SIM_KEY] = sim_score

            # how far is the match from the query location?
            data_dict[cs.I_DIST_KEY] = abs(qi - ci)
            data_dict[cs.NI_DIST_KEY] = abs(index - sub_idxs.index(ci))
            try:
                nearest_idx = sub_idxs[index + 1]
            except IndexError:
                nearest_idx = sub_idxs[index - 1]
            gt_T_wn = data_seq.get_true_pose(nearest_idx)
            gt_T_nw = np.linalg.inv(gt_T_wn)
            gt_T_n1 = gt_T_nw.dot(gt_T_w1)
            data_dict[cs.NEAR_DIST_KEY] = np.linalg.norm(gt_T_n1[:3, 3])

            if type(data_seq) is ColdDatasetSequence:
                # record the room ID and room type
                data_dict[cs.MROOM_KEY] = data_seq.get_room_id(ci)
                data_dict[cs.MROOM_TYPE_KEY] = data_seq.get_room_type(ci)
            if type(mdprc) is FabmapResultsContainer:
                data_dict[cs.NEW_PROB_KEY] = dist_result.conf_matrix[qi, qi]
        return data_dict

    result_data = []
    for step_dist_m, dist_results in mdprc.dist_experiments.items():
        if step_dist_m in step_exclude_list:
            continue
        for dist_result in dist_results:
            for idx in range(len(dist_result.matches)):
                data_dict = query_result_to_dict(step_dist_m, dist_result, idx)
                result_data.append(data_dict)

    return pd.DataFrame(result_data)


def print_stats_by_stepdist(results_df):
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None, 'display.width', 0):
        results_by_stepdist_df = results_df.groupby(cs.STEP_KEY).mean()
        print(results_by_stepdist_df)
        # for _, row in results_by_stepdist_df.mean().iterrows():
        #     print(row)


def plot_by_stepdist(results_dfs, df_names, stat_name, use_log=False,
                     legend_loc=None):
    plt.figure()
    steps = results_dfs[0][cs.STEP_KEY].unique()
    config_datasets = []
    for results_df, name in zip(results_dfs, df_names):
        results_df = results_df[results_df[stat_name].notnull()]
        # ignore samples deemed "new"
        # matched_results_df = results_df[results_df[cs.NEW_PROB_KEY] <
        #                                 results_df[cs.SIM_KEY]]
        matched_results_df = results_df
        print(name, len(matched_results_df), '/', len(results_df), 'are not new')
        dist_grp = matched_results_df.groupby(cs.STEP_KEY)
        stat_datasets = [list(data) for _, data in dist_grp[stat_name]]
        if len(stat_datasets) == 0:
            import pdb; pdb.set_trace()
        config_datasets.append(stat_datasets)

    plu.box_plot(results_dfs, df_names, cs.STEP_KEY, stat_name, use_log)
    # plu.violin_plot(config_datasets, df_names,
    #                 ds_labels=[str(ss) for ss in steps], log_scale=use_log,
    #                 legend_loc=legend_loc, add_legend=True)
    plt.xlabel(cs.STEP_KEY)
    plt.ylabel(stat_name)


def plot_xvsy_by_stepdist(results_dfs, df_names, x_name, y_name,
                          same_yrange=False):
    steps = results_dfs[0][cs.STEP_KEY].unique()
    sqrt = np.sqrt(len(steps))
    num_rows = int(np.ceil(sqrt))
    num_cols = int(np.ceil(len(steps) / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols)

    plt.ylabel(y_name)
    flat_axes = axes.flatten()
    for ax, step in zip(flat_axes, steps):
        ax.set_title(' '.join([cs.STEP_KEY, '=', str(step)]))
        # only show xticks on the bottom axis
        # if step != steps[-1]:
        for df, name in zip(results_dfs, df_names):
            dist_df = df.loc[df[cs.STEP_KEY] == step]
            # ignore samples deemed "new"
            dist_df = dist_df[dist_df[cs.NEW_PROB_KEY] < dist_df[cs.SIM_KEY]]

            ax.scatter(dist_df[x_name], dist_df[y_name], label=name)

    # set all plots to have the same ylim
    if same_yrange:
        ymax = max(ax.get_ylim()[1] for ax in flat_axes)
        for ax in flat_axes:
            ax.set_ylim(0, ymax)

    xmax = max(ax.get_xlim()[1] for ax in flat_axes)
    for ax in flat_axes:
        ax.set_xlim(0, xmax)
    # plt.tight_layout()
    flat_axes[0].set_ylabel(y_name)
    plt.xlabel(x_name)
    if len(df_names):
        plt.legend()


def plot_failures_by_stepdist(results_dfs, df_names):
    steps = results_dfs[0][cs.STEP_KEY].unique()
    plt.figure()
    # TODO make a bar plot for this
    for results_df, name in zip(results_dfs, df_names):
        failure_counts = []
        for step in steps:
            dist_df = results_df.loc[results_df[cs.STEP_KEY] == step]
            num_loc_failures = dist_df[cs.EST_DIST_KEY].isnull().sum()
            not_locfail_df = dist_df.loc[dist_df[cs.EST_DIST_KEY].notnull()]
            num_news = sum(not_locfail_df[cs.SIM_KEY] <
                           not_locfail_df[cs.NEW_PROB_KEY])
            num_news = 0
            failure_counts.append(num_loc_failures + num_news)
        plt.plot(steps, failure_counts, label=name)
    plt.title('failure counts vs. step dist')
    plt.legend()

# how often is the match in the top (1, 5, 10) physically closest images?
# how often is the true match in the top (1, 5, 10) highest-scoring images?
# let the "most similar" image be the determined match
# let the minimum distance image be the "true match"


def plot_column_comparison(results_dfs, df_names, x_key, y_key,
                           xlog=False, ylog=False):
    plt.figure()
    for results_df, name in zip(results_dfs, df_names):
        xs = results_df[x_key]
        if xlog:
            xs = np.log(xs)
        ys = results_df[y_key]
        if ylog:
            ys = np.log(ys)
        plt.scatter(xs, ys, label=name)

    if len(df_names) > 1:
        plt.legend()
    plt.xlabel(x_key)
    plt.ylabel(y_key)


def plot_confusion_matrices(results_df, name, key1, key2):
    plt.figure()

    steps = results_df[cs.STEP_KEY].unique()
    num_rows = int(np.ceil(np.sqrt(len(steps))))
    num_cols = int(np.ceil(len(steps) / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols)

    def get_ax(index, axes):
        if type(axes) is not np.ndarray:
            return axes
        if len(axes.shape) == 1:
            return axes[index]
        else:
            x_pos = index % axes.shape[0]
            y_pos = index // axes.shape[0]
            return axes[y_pos, x_pos]

    for ii, step in enumerate(steps):
        ax = get_ax(ii, axes)
        ax.set_title(' '.join([cs.STEP_KEY, '=', str(step)]))
        # only show xticks on the bottom axis
        if step != steps[-1]:
            ax.set_xticks([], [])
        dist_df = results_df.loc[results_df[cs.STEP_KEY] == step]
        conf_mat_df = pd.crosstab(dist_df[key1], dist_df[key2])
        # normalize it, since different rooms will have different frequencies
        ncf = conf_mat_df / conf_mat_df.sum(axis=0)
        # ax.scatter(dist_df[x_name], dist_df[y_name], label=name)
        mat = ax.matshow(ncf)
        mat.set_clim(vmin=0, vmax=1)
        ax.set_xticks(range(len(ncf.columns)), list(ncf.columns))
        ax.set_xlabel(ncf.columns.name)
        ax.set_yticks(range(len(ncf.index)), list(ncf.index))
        ax.set_ylabel(ncf.index.name)

    # plt.tight_layout()
    # get_ax(0, axes).set_ylabel(y_name)
    # plt.xlabel(x_name)
    # plt.legend()

    # conf_mat_df = pd.crosstab(results_df[key1], results_df[key2])
    # normalize it, since different rooms will have different frequencies
    # ncf = conf_mat_df / conf_mat_df.sum(axis=1)
    # plt.matshow(ncf)
    # plt.xticks(range(len(ncf.columns)), ncf.columns)
    # plt.xlabel(ncf.columns.name)
    # plt.yticks(range(len(ncf.index)), ncf.index)
    # plt.ylabel(ncf.index.name)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mat, cax=cbar_ax)
    plt.title(name)


def plot_roompred_rates(results_dfs, df_names, key1, key2):
    plt.figure()
    steps = results_dfs[0][cs.STEP_KEY].unique()

    for name, results_df in zip(df_names, results_dfs):
        errors = []
        for ii, step in enumerate(steps):
            dist_df = results_df.loc[results_df[cs.STEP_KEY] == step]
            conf_mat_df = pd.crosstab(dist_df[key1], dist_df[key2])
            num_correct = np.diag(conf_mat_df).sum()
            percent_correct = 100 * num_correct / len(dist_df)
            errors.append(percent_correct)
        plt.plot(steps, errors, label=name)
    plt.ylabel('Percent correct')
    plt.xlabel(cs.STEP_KEY)
    plt.legend()


def plot_newrates_vs_dist(results_dfs, df_names):
    plt.figure()
    steps = results_dfs[0][cs.STEP_KEY].unique()

    for results_df, name in zip(results_dfs, df_names):
        new_rates = []
        for ii, step in enumerate(steps):
            dist_df = results_df.loc[results_df[cs.STEP_KEY] == step]
            new_count = sum(dist_df[cs.SIM_KEY] < dist_df[cs.NEW_PROB_KEY])
            new_rates.append(100 * new_count / len(dist_df))
        plt.plot(steps, new_rates, '--o', label=name)
    plt.ylabel('Percent designated New')
    plt.xlabel(cs.STEP_KEY)
    plt.legend()


def plot_error_vs_sim_threshold(results_dfs, df_names, ykey, min_sim, max_sim,
                                nthreshs=20, sep_by_step=True):
    if sep_by_step:
        steps = results_dfs[0][cs.STEP_KEY].unique()
        sqrt = np.sqrt(len(steps))
        num_rows = int(np.ceil(sqrt))
        num_cols = int(np.ceil(len(steps) / num_rows))
        _, axes = plt.subplots(num_rows, num_cols)
        axes = axes.flatten()[:len(steps)]
    else:
        _, ax = plt.subplots()
        axes = [ax]

    key_label = 'Mean ' + ykey
    thresholds = np.linspace(min_sim, max_sim, nthreshs)

    handles = []
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax1color = colour_cycle[0]
    ax2color = colour_cycle[1]
    for ii, ax in enumerate(axes):
        ax.tick_params(axis='y', labelcolor=ax1color)
        ax2 = ax.twinx()
        ax2.tick_params(axis='y', labelcolor=ax2color)
        if sep_by_step:
            step = steps[ii]
            ax.set_title(' '.join([cs.STEP_KEY, '=', str(step)]))

        for name, results_df in zip(df_names, results_dfs):
            # if name == "FAB-MAP 2.0":
            #     import pdb; pdb.set_trace()
            if sep_by_step:
                df = results_df.loc[results_df[cs.STEP_KEY] == step]
            else:
                df = results_df

            # select only those elements that are above the threshold
            threshold_yvals = []
            omitteds = []
            for threshold in thresholds:
                mean_yval = df.loc[df[cs.SIM_KEY] > threshold][ykey].mean()
                threshold_yvals.append(mean_yval)
                num_omitted = len(df.loc[df[cs.SIM_KEY] <= threshold])
                frac_omitted = num_omitted / len(df)
                omitteds.append(100 * frac_omitted)

            err_plot = ax.plot(thresholds, threshold_yvals, color=ax1color,
                               label=key_label)[0]
            handles.append(err_plot)

            percent_label = '% queries designated "New"'
            percent_plot = ax2.plot(thresholds, omitteds, color=ax2color,
                                    # color=err_plot.get_color(), linestyle='--',
                                    label=percent_label)[0]
            handles.append(percent_plot)

        ax2.set_ylabel(percent_label)
        # fabmap on kitti says 12.7% are new
        err_line = ax.axhline(252.88, zorder=-1, color=ax1color,
                              linestyle='--',
                              label='FAB-MAP 2.0 mean pose error (m)')
        handles.append(err_line)
        percent_line = ax2.axhline(12.69, zorder=-1, color=ax2color,
                                   linestyle='--',
                                   label='% new according to FAB-MAP 2.0')
        handles.append(percent_line)
        # if not sep_by_step:
        #     ax.legend(handles=[plot, plot2, line1])


    # # set all plots to have the same ylim
    # ymax = max(ax.get_ylim()[1] for ax in axes)
    # for ax in axes:
    #     ax.set_ylim(0, ymax)

    # plt.tight_layout()
    axes[0].set_ylabel(key_label)
    axes[0].set_xlabel('Similarity score threshold')
    if not sep_by_step:
        handles = [mlines.Line2D([], [], color='black', linestyle='--',
                                 label='FAB-MAP 2.0'),
                   mlines.Line2D([], [], color='black',
                                 label='Object Landmarks')]
        # if we're plotting many small plots, they're too small for the legend
        plt.legend(handles=handles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs="+",
                        help='A pickle file containing saved results.')
    parser.add_argument('-n', '--name', action='append',
                        help='Labels for the data from each pickle file.')
    parser.add_argument('--ex', action='append', type=float, help='\
Step distance to exclude.  More than one may be specified.')
    parser.add_argument('--simthresh', type=float, help='\
If provided, ignore entries below this similarity threshold.')
    args = parser.parse_args()
    results_list = []
    if args.name:
        names = args.name
    else:
        names = []
    for data_file in args.data:
        data_name = Path(data_file).stem
        with open(data_file, 'rb') as ff:
            stored_results = pickle.load(ff)
        try:
            results_list += stored_results
            if len(names) != len(args.data):
                names += [data_name] * len(stored_results)
        except TypeError:
            # sometimes we might have forgotten to pickle a single element as a
            # list...
            results_list += [stored_results]
            if len(names) != len(args.data):
                names.append(data_name)

    data_types = set([rr.data_seq_type for rr in results_list])
    if len(data_types) > 1:
        raise ValueError('\
User provided data results from multiple types of data source!')
    data_type = data_types.pop()
    # each MultiDistPrResults object in the list corresponds to one sequnce
    # and config.  Each should be handled separately.
    # TODO figure out naming...
    fabmap_results = []
    fabmap_names = []
    mypr_results = []
    mypr_names = []
    for rr, nn in zip(results_list, names):
        if type(rr) is FabmapResultsContainer:
            fabmap_results.append(rr)
            fabmap_names.append(nn)
        if type(rr) is MultiDistPrResultsContainer:
            mypr_results.append(rr)
            mypr_names.append(nn)

    results_list = fabmap_results + mypr_results
    names = fabmap_names + mypr_names
    step_exclude_list = []
    if args.ex is not None:
        step_exclude_list = args.ex
    dfs = [get_multi_dist_results_df(mdprc, step_exclude_list, args.simthresh)
           for mdprc in results_list]
    for df, name in zip(dfs, names):
        print('stats for', name)
        print_stats_by_stepdist(df)

    # plot_by_stepdist(dfs, names, cs.NEAR_DIST_KEY)
    plot_column_comparison(dfs, names, cs.SIM_KEY, cs.DIST_KEY)
    plot_xvsy_by_stepdist(dfs, names, cs.SIM_KEY, cs.DIST_KEY)
    if not args.simthresh:
        # plot_error_vs_sim_threshold(dfs, names, cs.ABS_POS_ERR_KEY, 0, 0.3)
        # plot_error_vs_sim_threshold(dfs, names, cs.ABS_POS_ERR_KEY, 0, 0.3,
        #                             sep_by_step=False)
        plot_error_vs_sim_threshold(dfs, names, cs.SYM_ERR_KEY, 0.05, 0.2)
        plot_error_vs_sim_threshold(dfs, names, cs.SYM_ERR_KEY, 0.05, 0.2,
                                    sep_by_step=False)
    else:
        plot_newrates_vs_dist(dfs, names)

    # plot_by_stepdist(dfs, names, cs.DIST_KEY)
    # plot_by_stepdist(dfs, names, cs.ROT_ERR_KEY)
    if data_type is ColdDatasetSequence:
        plot_by_stepdist(dfs, names, cs.DIST_KEY)
        # plot_by_stepdist(dfs, names, cs.ROT_ERR_KEY)
        # plot_by_stepdist(dfs, names, cs.COS_ERR_KEY)
        plot_failures_by_stepdist(dfs, names)
        # plot_xvsy_by_stepdist(dfs, names, cs.DIST_KEY, cs.ROT_ERR_KEY)
        # plot_xvsy_by_stepdist(dfs, names, cs.DIST_KEY, cs.COS_ERR_KEY)
        # # plot the confusion matrices
        plot_roompred_rates(dfs, names, cs.QROOM_KEY, cs.MROOM_KEY)
        # for df, name in zip(dfs, names):
        #     plot_confusion_matrices(df, name, cs.QROOM_KEY, cs.MROOM_KEY)
        # TODO whatever else we think might be useful...something to do with
        # new probs?

    elif data_type is KittiSequence:
        # plots for everything in a KITTI sequence
        plot_failures_by_stepdist(dfs, names)
        plot_by_stepdist(dfs, names, cs.SIM_KEY)
        plot_by_stepdist(dfs, names, cs.SYM_ERR_KEY, use_log=True,
                         legend_loc='lower right')
        # plot_by_stepdist(dfs, names, cs.SYM_ERR_KEY, legend_loc='upper left')
        # plot_by_stepdist(dfs, names, cs.DIST_KEY)
        # plot_xvsy_by_stepdist(dfs, names, cs.DIST_KEY, cs.ABS_POS_ERR_KEY)

        # our-method-specific plots
        plot_column_comparison(dfs, names, cs.SIM_KEY, cs.SYM_ERR_KEY)
        plot_xvsy_by_stepdist(dfs, names, cs.SIM_KEY, cs.SYM_ERR_KEY)
        # # fabmap-specific-plots
        # plot_column_comparison(dfs[:len(fabmap_results)], ['fabmap', 'fabmap all'],
        #                        NEW_PROB_KEY, ABS_POS_ERR_KEY)

    plt.show()


if __name__ == "__main__":
    main()
