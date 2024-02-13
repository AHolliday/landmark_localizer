#!/usr/bin/env python

import os
import yaml
import pickle
import argparse
from sequence_experiment import SequenceExperimentResult
from landmark_localizer import experimentUtils as expu


parser = argparse.ArgumentParser()
parser.add_argument('pickle')
parser.add_argument('filterConfig')
parser.add_argument('--prefix', default='filt_', help='A prefix for the output pickle file.')
args = parser.parse_args()

with open(args.pickle, 'r') as f:
    results = pickle.load(f)
with open(args.filterConfig, 'r') as f:
    fltrCfg = yaml.load(f)
matchingResults = [x for x in results if expu.doesConfigMatchTarget(x.experiment_config,
                                                              fltrCfg)]
newPickle = args.prefix + os.path.basename(args.pickle)
with open(newPickle, 'w') as f:
    pickle.dump(matchingResults, f)
