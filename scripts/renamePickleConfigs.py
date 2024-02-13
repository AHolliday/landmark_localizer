#!/usr/bin/env python

import os
import yaml
import pickle
import argparse
from realWorldExperiment import Result
from landmark_localizer import experimentUtils as expu


parser = argparse.ArgumentParser()
parser.add_argument('pickle')
args = parser.parse_args()

with open(args.pickle, 'r') as f:
    baseCfg, results = pickle.load(f)
newResults = []
for cfg, resultSet in results:
    print('Config name is', cfg['name'])
    newName = eval(input('New name:'))
    cfg['name'] = newName

for cfg, _ in results:
    print(cfg['name'])

dirname, basename = os.path.split(args.pickle)
newPickle = os.path.join(dirname, 'rename_' + basename)
with open(newPickle, 'w') as f:
    pickle.dump((baseCfg, results), f)
