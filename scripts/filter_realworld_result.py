#!/usr/bin/env python

import os
import yaml
import pickle
import argparse
from realWorldExperiment import Result
from landmark_localizer import experimentUtils as expu


parser = argparse.ArgumentParser()
parser.add_argument('pickle')
parser.add_argument('filterConfig')
parser.add_argument('--prefix', default='best_', help='A prefix for the output pickle file.')
args = parser.parse_args()

with open(args.pickle, 'r') as f:
    baseCfg, results = pickle.load(f)
with open(args.filterConfig, 'r') as f:
    fltrCfg = yaml.load(f)
matchingResults = [x for x in results if expu.doesConfigMatchTarget(x[0], fltrCfg)]
newPickle = args.prefix + os.path.basename(args.pickle)
with open(newPickle, 'w') as f:
    if len(matchingResults) == 1:
        baseCfg = dict(matchingResults[0][0])
    pickle.dump((baseCfg, matchingResults), f)
