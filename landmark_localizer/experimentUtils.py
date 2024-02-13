import argparse
import yaml
import os
import numpy as np
import torch

from . import ObjectExtractor as oe


# box-aggregation keywords
NMS = 'nms'
AVG = 'avg'
NO_AGG = 'none'

# full test functions to be called from main
SS_STR = 'selective_search'
VGG_STR = 'vgg16'
SIFT_STR = 'sift'
LIFT_STR = 'lift'
ORB_STR = 'orb'
RANDOM_STR = 'random'
PATCH_STR = 'patch'
D2_STR = 'd2net'

FIXED_NET_TYPES = [oe.ALEX_KEY] + oe.RESNET_KEYS + oe.DENSE_KEYS + oe.VGG_KEYS

RANGE_KEYS = set(['min', 'max', 'step'])


def get_extractor_type(config):
    return config['extractor']['type']


def get_feature(config):
    return str(sorted(config['extractor']['oeKwargs']['featureBlobNames']))


def getMatchingConfigIdxs(targetConfig, configs):
    matches = [doesConfigMatchTarget(targetConfig, c) for c in configs]
    return np.where(matches)


def doesConfigMatchTarget(config, targetConfig):
    for targetKey, targetValue in targetConfig.items():
        if targetKey not in config:
            continue
        value = config[targetKey]

        if isinstance(targetValue, dict):
            if set(targetValue.keys()) == RANGE_KEYS:
                targetValue = _expandRangeConfig(targetValue)
            else:
                if doesConfigMatchTarget(value, targetValue):
                    continue
                else:
                    return False

        # it might have been a dict and become a list in the previous step
        if isinstance(targetValue, list):
            # config's value passes if it's in the target's list of values
            if value in targetValue:
                continue
            else:
                return False

        # base case
        if value != targetValue:
            return False
    return True


def expandConfigToParamGrid(baseConfig):
    namesWithConfigs = _expandConfigToParamGrid_helper(baseConfig)
    for name, cfg in namesWithConfigs:
        if 'name' not in cfg:
            cfg['name'] = ''
        if name:
            cfg['name'] += '_' + name
    _, configs = list(zip(*namesWithConfigs))
    return configs


def _expandConfigToParamGrid_helper(baseConfig):
    configGrid = [('', baseConfig)]
    # if element is list, create one with each element.
    for key in sorted(baseConfig):
        value = baseConfig[key]
        newValuesList = None
        isList = isinstance(value, list)
        isNpRange = isinstance(value, dict) and set(value.keys()) == RANGE_KEYS

        if isList or isNpRange:
            # base case.
            if isList:
                # create one config with each element
                newValuesList = value
            elif isNpRange:
                # create one with each element in the range
                newValuesList = _expandRangeConfig(value)
            # generate names
            newValuesNames = [key + '=' + str(v) for v in newValuesList]
            newValuesList = list(zip(newValuesNames, newValuesList))

        elif isinstance(value, dict):
            # recurse and get sub-grids
            newValuesList = _expandConfigToParamGrid_helper(value)

        if newValuesList is not None:
            # add the new values to the grid
            newConfigGrid = []
            for basename, config in configGrid:
                for elemname, elem in newValuesList:
                    elemConfig = config.copy()
                    elemConfig[key] = elem
                    if basename and elemname:
                        # they're both non-empty, so concat nicely.
                        elemname = basename + '_' + elemname
                    else:
                        # one or both are empty, so just take what isn't.
                        elemname = basename + elemname
                    newConfigGrid.append((elemname, elemConfig))
            configGrid = newConfigGrid

    return configGrid


def _expandRangeConfig(rangeCfg):
    range_ = np.arange(rangeCfg['min'], rangeCfg['max'], rangeCfg['step'])
    # return default python types for compatibility with numpy.
    range_ = list(map(np.asscalar, range_))
    return range_


def getObjectExtractorFromConfig(config, device=None):
    if 'extractor' in config:
        config = config['extractor']

    if not config['type']:
        raise ValueError("No object extractor type specified!")

    # get generic keyword arguments
    oeKwargs = {}
    if 'oeKwargs' in config:
        oeKwargs = config['oeKwargs']

    # get functions needed by some extractors
    if 'filter' in config:
        # add filtering configuration to object-extractor arguments
        filterParams = config['filter']
        oeKwargs['iouThreshold'] = filterParams['iouThreshold']
        aggFunKey = filterParams['type']
        print(aggFunKey)
        oeKwargs['aggBoxFunction'] = getAggBoxFunction(aggFunKey)

    # now check each possible value of the extractor type
    if config['type'] == RANDOM_STR:
        objEngine = oe.RandomExtractor()

    elif config['type'] == LIFT_STR:
        objEngine = oe.LiftWrapperExtractor(**config['oeKwargs'])

    elif config['type'] == SIFT_STR:
        # instantiate sift wrapper here, as it doesn't use other config
        siftKwargs = {}
        if 'siftParams' in config:
            siftKwargs = config['siftParams']
        if 'imageInputSize' in oeKwargs:
            objEngine = oe.SiftWrapperExtractor(siftKwargs,
                                                oeKwargs['imageInputSize'])
        else:
            objEngine = oe.SiftWrapperExtractor(siftKwargs)

    elif config['type'] == ORB_STR:
        orbKwargs = {}
        if 'orbParams' in config:
            orbKwargs = config['orbParams']
        if 'imageInputSize' in oeKwargs:
            objEngine = oe.OrbWrapperExtractor(orbKwargs,
                                               oeKwargs['imageInputSize'])
        else:
            objEngine = oe.OrbWrapperExtractor(orbKwargs)

    elif config['type'] == D2_STR:
        objEngine = oe.D2NetReader(**oeKwargs)

    elif config['type'] in FIXED_NET_TYPES:
        objEngine = oe.FixedCnnPatchExtractor(config['type'], device=device,
                                              **oeKwargs)
    else:
        raise ValueError('config type "' + config['type'] +
                         '" not recognized!')

    return objEngine


def getAggBoxFunction(strkey):
    if strkey == NMS:
        return oe.nonMaxSuppressionFast
    elif strkey == AVG:
        return oe.averageBoxesFast
    else:
        return None


def getConfigsFromArgs(args):
    configs = []
    if args.cfgFiles:
        for cfgFile in args.cfgFiles:
            # read in the provided configuration file.
            with open(cfgFile, 'r') as ff:
                config = yaml.load(ff)
                # add the basename of the file as a name, sans type suffix.
                if 'name' not in config:
                    name = os.path.basename(cfgFile).rpartition('.')[0]
                    config['name'] = name
                configs.append(config)

    else:
        # construct a config dict from the user's provided arguments.
        config = {}
        localization = {}
        localization['planar'] = args.planar
        localization['metric'] = args.metric
        if args.ransac is not None:
            localization['ransacThreshold'] = args.ransac
        config['localization'] = localization

        extractor = {}
        extractor['type'] = args.oeType
        extractor['oeKwargs'] = {}
        if args.oeType is not SIFT_STR:
            extractor['oeKwargs']['featureBlobNames'] = args.layer
        if args.size:
            extractor['oeKwargs']['imageInputSize'] = args.size

        config['extractor'] = extractor

        # give it a name
        if args.name:
            config['name'] = args.name
        else:
            config['name'] = args.oeType
        configs = [config]

    return configs


def addCommonArgs(parser=argparse.ArgumentParser()):
    # method parameters
    # the user can only pick one extractor type.
    parser.add_argument('--cfg', action='append', dest='cfgFiles', help='\
    A configuration file for the experiment.')
    oeGroup = parser.add_mutually_exclusive_group()
    oeGroup.add_argument('--ss', action='store_const', dest='oeType',
                         const=SS_STR, help='\
    Use selective-search object proposals and CNN features.')
    # oeGroup.add_argument('--resnet', action='store_const', dest='oeType',
    #                      const=oe.RESNET_50_KEY, help='Use ResNet-50 features.')
    oeGroup.add_argument('--sift', action='store_const', dest='oeType',
                         const=SIFT_STR, help='\
    Use SIFT point proposals and features.')
    oeGroup.add_argument('--random', action='store_const', dest='oeType',
                         const=RANDOM_STR, help='\
    Use random boxes and features.  This won\'t be very useful.')

    # parameters for extractor behaviour
    parser.add_argument('--layer',
                        help='layer from which to extract features.')
    parser.add_argument('--iou', type=float, default=0.45, help='Threshold \
    on intersection-over-union for grouping proposed boxes.')

    # context parameters
    parser.add_argument('--metric', default='cosine',
                        help='string distance metric to use for matching.')
    parser.add_argument('--planar', action='store_true', help="\
    Treat the environment as planar.")
    parser.add_argument('--ransac', type=float, help='\
    if provided, localize using ransac with this threshold.')
    parser.add_argument('--name', help='\
    A name for this experiment.')
    parser.add_argument('--size', type=int,
                        help='Size of the image\'s small axis.')
    device_grp = parser.add_mutually_exclusive_group()
    device_grp.add_argument('--cpu', action='store_true',
                            help='if provided, run torch on the CPU')
    device_grp.add_argument('--gpu', type=int, default=0,
                            help='ID of the GPU on which to run torch.')
    return parser


def getDeviceFromArgs(args):
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu))
    return device


def main():
    """
    Tests the expansion of configuration files into parameter grids.
    """
    parser = addCommonArgs()
    args = parser.parse_args()
    baseCfgs = getConfigsFromArgs(args)
    for baseCfg in baseCfgs:
        cfgs = expandConfigToParamGrid(baseCfg)
        for cfg in cfgs:
            print(cfg['name'])
            print(cfg)


if __name__ == "__main__":
    main()
