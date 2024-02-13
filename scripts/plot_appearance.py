import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from landmark_localizer import constants as cs
from landmark_localizer import plot_utils as pu


import matplotlib
matplotlib.rc('font', size=15)


def plot_accuracy(scores_df, error_key):
    types = set(''.join(cc for cc in tt if not cc.isdigit())
                    for tt in scores_df['type'].unique())
    plt_bottom = scores_df[error_key].min() - 1
    plt_top = scores_df[error_key].max() + 1

    for type in types:
        type_df = scores_df[scores_df['type'].str.contains(type)]
        if type not in cs.LAYER_ORDERS:
            continue
        layer_order = [ll for ll in cs.LAYER_ORDERS[type]
                       if ll in type_df['feature'].unique()]
        errors_by_subtype = []

        # this preserves the ordering of the types, unlike 'groupby'
        subtypes = type_df['type'].unique()
        for subtype in subtypes:
            sub_df = type_df[type_df['type'] == subtype]
            errors = []
            for ll in layer_order:
                row = sub_df[sub_df['feature'] == ll]
                if len(row) > 0:
                    errors.append(row.iloc[0][error_key])
                else:
                    errors.append(None)
            errors_by_subtype.append(errors)
        pu.bar_chart(errors_by_subtype, subtypes, layer_order)
        # plt.ylim(plt_bottom, plt_top)
        plt.ylim(0, 80)
        plt.legend()
        plt.ylabel(error_key)
        # subtypes.append(subtype)


def plot_error_vs_size(scores_df, error_key, separate_subtypes=True,
                       label_points=False, filter_best=False):
    # plot all vs. sizes
    # add sizes to dataframe
    sizes = []
    if 'size' not in scores_df.columns:
        for _, row in scores_df.iterrows():
            if row['type'] in cs.NET_LAYER_SIZES:
                sizes.append(cs.NET_LAYER_SIZES[row['type']][row['feature']])
            elif row['type'].lower() == 'sift':
                sizes.append(128)
        scores_df['size'] = sizes
    plt.figure()

    netarchs = ['alexnet', 'resnet', 'vgg', 'densenet']
    point_styles = ["o", "v", "X", "s", "d", "^", "x", "+", "*", "P"]
    netarch_counts = defaultdict(int)

    blacklist = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg13',
                 'vgg16', 'vgg19', 'alexnet']
    if filter_best:
        scores_df = scores_df[~scores_df['type'].isin(blacklist)]

    if separate_subtypes:
        subtypes = scores_df['type'].unique()
        for subtype in subtypes:
            stdf = scores_df[scores_df['type'] == subtype]
            netarch = ''.join(cc for cc in subtype if not cc.isdigit())
            colour_idx = netarchs.index(netarch)
            colour = 'C{}'.format(colour_idx)
            style = point_styles[netarch_counts[netarch]]
            netarch_counts[netarch] += 1
            plt.scatter(stdf['size'], stdf[error_key], label=subtype,
                        marker=style, color=colour)
            if label_points:
                for _, row in stdf.iterrows():
                    loc = (row['size'], row[error_key])
                    plt.annotate(row['feature'], loc)
    else:
        for netarch in netarchs:
            tdf = scores_df[scores_df['type'].str.startswith(netarch)]
            plt.scatter(tdf['size'], tdf[error_key], label=netarch)

    plt.ylim(0, 80)
    plt.legend()
    plt.xlabel('feature sizes')
    plt.ylabel(error_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='+', help='\
Paths to csv files containing results to plot.')
    args = parser.parse_args()
    all_dfs = [pd.read_csv(csv) for csv in args.csv]
    df = pd.concat(all_dfs).reset_index(drop=True)
    error_key = 'symmetric error'
    print('min error:')
    print(df.iloc[df[error_key].idxmin()])
    # print('max iou:')
    # print(df.iloc[df['mean iou'].idxmax()])
    plot_accuracy(df, error_key)
    # plot_accuracy(df, 'mean iou')
    plot_error_vs_size(df, error_key, separate_subtypes=False)
    plot_error_vs_size(df, error_key, label_points=True, filter_best=True)
    # plot_error_vs_size(df, 'mean iou')
    plt.show()

if __name__ == "__main__":
    main()
