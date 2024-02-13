import argparse
import pickle
import torch
from tqdm import tqdm
from pathlib import Path
from itertools import combinations

from landmark_localizer import experimentUtils as expu
from landmark_localizer import torchUtils as tchu
from landmark_localizer import localization as loc


class AppearanceImagePairResult:
    def __init__(self, cfg, ipath, jpath, dists, iboxes, jboxes,
                 ipoints=None, jpoints=None):
        self.config = cfg
        self.ipath = ipath
        self.jpath = jpath
        self.dists = dists
        self.iboxes = iboxes
        self.jboxes = jboxes
        self._ipoints = ipoints
        self._jpoints = jpoints

    @property
    def point_matches(self):
        if None not in [self._ipoints, self._jpoints]:
            return list(zip(self._ipoints, self._jpoints))
        else:
            def center(bb): return (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            ipoints = [center(bb) for bb in self.iboxes]
            jpoints = [center(bb) for bb in self.jboxes]
            return list(zip(ipoints, jpoints))
        #
        # haveSubFeatures
        #


def appearance_experiment(config, dataset_path, device, subsample=None):
    # build extractor
    extractor = expu.getObjectExtractorFromConfig(config, device)
    # get metric, sub-feature metric from config
    metric = config['localization']['metric']
    subFeatureMetric = config['localization'].get('subFeatureMetric')

    # build subsampled list of images
    image_paths = [pp for pp in Path(dataset_path).iterdir()
                   if pp.suffix in ['.jpg', '.png']]
    if subsample:
        image_paths = image_paths[::subsample]

    # extract features from all images
    scenes = [extractor.detectAndComputeScene(str(pp)) for pp in image_paths]
    # move all the features to the specified device at once

    all_results = []
    combs = list(combinations(range(len(scenes)), 2))
    for ii, jj in tqdm(combs):
        # match features
        featsi = torch.tensor(scenes[ii].descriptors, device=device)
        featsj = torch.tensor(scenes[jj].descriptors, device=device)
        matches, dists = tchu.matchByBruteForce(featsi, featsj, metric)
        mObjs1, mObjs2 = list(zip(*[(scenes[ii].objects[kk],
                                     scenes[jj].objects[ll])
                                    for kk, ll in matches]))
        haveSubFeatures = any([len(oo.getSubFeatures()) > 0
                               for oo in mObjs1 + mObjs2])
        pts1, pts2 = [], []
        if haveSubFeatures:
            matchSubFeaturePts, dists = loc.matchSubFeatures(mObjs1, mObjs2,
                                                             subFeatureMetric,
                                                             crossCheck=True)
            if len(matchSubFeaturePts) > 0:
                subPoints1, subPoints2 = list(zip(*matchSubFeaturePts))
                pts1 += subPoints1
                pts2 += subPoints2
        if len(pts1) == 0 and len(pts2) == 0:
            pts1, pts2 = None, None
        result = AppearanceImagePairResult(config, image_paths[ii],
                                           image_paths[jj], dists,
                                           [mo.box for mo in mObjs1],
                                           [mo.box for mo in mObjs2],
                                           pts1, pts2)
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', nargs='+', help='\
A directory containing images to compare against one another.')
    parser.add_argument('--sub', type=int, help='Subsampling factor.')
    parser = expu.addCommonArgs(parser)
    args = parser.parse_args()
    device = expu.getDeviceFromArgs(args)

    all_base_cfgs = expu.getConfigsFromArgs(args)
    for datadir in args.datadir:
        for base_cfg, cfg_path in zip(all_base_cfgs, args.cfgFiles):
            cfg_results = []
            for cfg in expu.expandConfigToParamGrid(base_cfg):
                results = appearance_experiment(cfg, datadir, device, args.sub)
                cfg_results.append((cfg, results))
            cfg_name = Path(cfg_path).stem
            out_path = Path(datadir) / (cfg_name + '.pkl')
            with out_path.open('wb') as ff:
                pickle.dump(cfg_results, ff)


if __name__ == "__main__":
    main()
