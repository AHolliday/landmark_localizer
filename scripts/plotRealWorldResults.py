#!/usr/bin/env python

import os
import yaml
import argparse
import pickle
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from itertools import cycle
from collections import defaultdict

from realWorldExperiment import Result
from landmark_localizer import transformations as tfs
from landmark_localizer import plot_utils as pu
from landmark_localizer import experimentUtils as expu
from landmark_localizer import geometryUtils as geo
from landmark_localizer import localization as loc
from landmark_localizer import constants as consts


matplotlib.rc('font', size=12)


STYLES = ["o", "v", "X", "s", "d", "^", "x", "+", "*", "P"]
MAX_ERROR = 1e7


def generateColours(numToGenerate, colourmapName='gist_rainbow'):
    cmap = plt.get_cmap(colourmapName)
    space = np.linspace(0, 1, numToGenerate)
    colours = np.array([cmap(sample) for sample in space])
    return colours


def groupByImagePair(ungroupedResultsList):
    img_pair_results = defaultdict(list)
    # group by dataset and image pair
    for result in ungroupedResultsList:
        img_pair_results[result.data_pair_name].append(result)
    return img_pair_results


def plotManyResults(results_with_configs, config_names=None,
                    showScaleBars=False, showNames=False, plotLines=False,
                    plotFilename=None):
    styleCycle = cycle(STYLES)
    namedBadnesses = []

    for ii, (config, ungroupedResultsList) in enumerate(results_with_configs):
        img_pair_results = groupByImagePair(ungroupedResultsList)

        # format errors for ranking
        style = next(styleCycle)
        if config is None:
            plotName = 'Ground truth'
        elif config_names:
            plotName = config_names[ii]
        else:
            plotName = config['name']

        errors = []
        ybars = []
        scaleChangeLists = []
        failureCounts = []
        for data_name, resultsList in img_pair_results.items():
            scaleChanges = resultsList[0].gt_scale_changes()
            if np.median(scaleChanges) > 8:
                # ignore image pairs with scale change greater than 8.
                continue
            numFailures = len([rr for rr in resultsList if rr.is_failure])
            failureCounts.append(numFailures)
            scaleChangeLists.append(resultsList[0].gt_scale_changes())
            data_errors = [rr.error() if not rr.is_failure else MAX_ERROR
                           for rr in resultsList]
            if len(data_errors) > 0:
                data_errors = np.log10(data_errors)
                errors.append(np.mean(data_errors))
                if len(resultsList) > 1:
                    # plot error bars
                    ybars.append(np.std(data_errors))
                else:
                    ybars.append(0)

        logErrors = errors
        medScaleChanges = [np.median(scs) for scs in scaleChangeLists]
        bottomBars = [np.median(scs) - min(scs) for scs in scaleChangeLists]
        topBars = [max(scs) - np.median(scs) for scs in scaleChangeLists]
        stdDevs = [np.std(scs) for scs in scaleChangeLists]
        color_code = 'C{}'.format(ii)

        # plt.figure(1
        if showScaleBars:
            plt.errorbar(medScaleChanges, logErrors,
                         xerr=(bottomBars, topBars), fmt=style, capsize=5,
                         label=plotName, color=color_code)
        else:
            # if np.any([bb > 0 for bb in ybars]):
            #     plt.errorbar(medScaleChanges, logErrors, yerr=ybars, fmt=style,
            #                  capsize=5, label=plotName, color=color_code)
            # else:
            plt.scatter(medScaleChanges, logErrors, marker=style,
                        label=plotName, color=color_code)

        # plot a best-fit curve for this config
        if plotLines:
            fitXs, fitYs = pu.get_plottable_fit_curve(medScaleChanges,
                                                      logErrors, 1)
            plt.plot(fitXs, fitYs)
        if showNames:
            # locations = [r.data_pair_name for r in resultsList
            #              if np.median(r.gt_scaled]
            for i, result in enumerate(resultsList):
                location = (medScaleChanges[i], logErrors[i])
                plt.annotate(result.data_pair_name, location)
        plt.xlabel('Median ground-truth scale change')
        plt.ylabel('Logarithmic STE')

        # plt.figure(2)
        # xPos = np.array(medScaleChanges)
        # # plt.plot(medScaleChanges, failureCounts, label=plotName)
        # plt.bar(xPos + ii * 0.1, failureCounts, 0.1, label=plotName,
        #         tick_label=list(map('{:0.1f}'.format, medScaleChanges)))

        # penalize both failures and errors
        badness = np.mean(logErrors)
        namedBadnesses.append((plotName, badness, config))

    print('from best to worst')
    sortedNamedBadnesses = sorted(namedBadnesses, key=lambda p: p[1])
    for name, badness, _ in sortedNamedBadnesses:
        print(name, 'badness:', badness)

    # display the plot
    # plt.figure(2)
    # plt.legend()
    # plt.show()

    plt.figure(1)
    # show the max error on the plot
    plt.axhline(np.log10(MAX_ERROR), color='black', zorder=-1)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    if plotFilename:
        plt.savefig(plotFilename, bbox_inches='tight')
    else:
        plt.show()


def countWinners(results_with_configs):
    errorsByConfig = []
    failuresByConfig = []
    for config, resultsList in results_with_configs:
        if config is None:
            plotName = 'Ground truth'
        else:
            plotName = config['name']
        errors = [r.error() for r in resultsList]
        failuresByConfig.append((plotName, errors.count(None)))
        for i, e in enumerate(errors):
            if e is None:
                errors[i] = np.inf
        errorsByConfig.append((plotName, errors))

    # Determine in how many cases the config 'wins'
    names, errorLists = list(zip(*errorsByConfig))
    errorMatrix = np.array(errorLists)
    winCounts = np.zeros(errorMatrix.shape[0], dtype=int)
    for i in range(errorMatrix.shape[1]):
        winnerIdx = np.argmin(errorMatrix[:, i])
        winCounts[winnerIdx] += 1
    counts_with_names = list(zip(winCounts, names))
    for count, name in counts_with_names:
        print(name, 'wins', count)
    print('winning method:', names[np.argmax(winCounts)])

    # # print the number of failures for each config
    # for name, count in failuresByConfig:
    #     if count != 0:
    #         print name, 'failed', count, 'times'
    # names, failureCounts = zip(*failuresByConfig)
    # print 'most failures:', names[np.argmax(failureCounts)]


def countFailures(results_with_configs):
    failuresByConfig = []
    for config, resultsList in results_with_configs:
        if config is None:
            plotName = 'Ground truth'
        else:
            plotName = config['name']
        errors = [r.error() for r in resultsList]
        failuresByConfig.append((plotName, errors.count(None)))
    for name, count in failuresByConfig:
        if count != 0:
            print(name, 'failed', count, 'times')
    names, failureCounts = list(zip(*failuresByConfig))
    print('most failures:', names[np.argmax(failureCounts)])


def main():
    parser = argparse.ArgumentParser()

    # the user can run a new map experiment, or just analyze results from an
    # old one.  If they're not loading an old experiment, they can dump the
    # results of the new one.
    parser.add_argument('pickle', nargs="+", help="\
    Read the pickled map test results from the provided file instead of \
    running a map test from scratch.")
    parser.add_argument('-s', '--save', help='\
    If provided, save the generated figure with this filename.')
    parser.add_argument('--filterCfg', help="\
Only display configurations matching this filter config.")
    parser.add_argument('--bars', action='store_true', help="\
Display bars showing the range of scale changes.")
    parser.add_argument('-n', '--name', action='append',
                        help='Labels for the data from each pickle file.')
    parser.add_argument('--pairnames', action='store_true', help="\
Display names over each point indicating which data pair it comes from.")
    parser.add_argument('--lines', action='store_true', help="\
Plot best-fit lines for each config.")
    # parser = expu.addCommonArgs(parser)
    args = parser.parse_args()

    allResults = []
    for pklFilename in args.pickle:
        with open(pklFilename, 'rb') as f:
            baseCfg, resultsWConfigs = pickle.load(f)
            allResults += resultsWConfigs
    if args.name:
        cfg_names = args.name
    else:
        cfg_names = None
    if args.filterCfg:
        with open(args.filterCfg, 'r') as f:
            filterCfg = yaml.load(f)
        matchResults = [x for x in allResults
                        if expu.doesConfigMatchTarget(x[0], filterCfg)]
        plotResults = matchResults
    else:
        plotResults = allResults
    plotManyResults(plotResults, cfg_names, args.bars, args.pairnames,
                    args.lines, plotFilename=args.save)
    # countWinners(plotResults)
    # countFailures(plotResults)


if __name__ == "__main__":
    main()
