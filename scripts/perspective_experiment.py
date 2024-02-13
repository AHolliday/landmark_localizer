import argparse
import pickle
import cv2
import torch
from tqdm import tqdm
from pathlib import Path
from itertools import combinations
from scipy import io

from sequence_experiment import estimate_transform
from landmark_localizer import geometryUtils as geo
from landmark_localizer import experimentUtils as expu
from landmark_localizer import torchUtils as tchu
from landmark_localizer import localization as loc


class PerspectiveDataset:
    def __init__(self, data_dir, calib_path, subsample_step):
        self.data_dir = data_dir
        self.subsample_step = subsample_step
        self.calib_dict = io.loadmat(calib_path)

    @property
    def K_px(self):
        return self.calib_dict['KK']

    def get_all_image_keys(self):
        # note that we only use images 1 through 55, as these represent the
        # first, closest arc
        return list(range(1, 56, self.subsample_step))

    def get_all_image_paths(self):
        keys = self.get_all_image_keys()

        paths = []
        for path in Path(self.data_dir).glob('*_09.bmp'):
            pose_idx = int(path.stem.partition('Img')[2].split('_')[0])
            if pose_idx in keys:
                paths.append(str(path))
        return paths

    def get_true_pose(self, key):
        RR = self.calib_dict['Rc_{}'.format(key)]
        TT = self.calib_dict['Tc_{}'.format(key)]
        return geo.buildTfMatrix(RR, TT)

    def get_image(self, image_key):
        filename = 'Img{0:0=3d}_09.bmp'.format(image_key)
        path = str(Path(self.data_dir) / filename)
        return cv2.imread(path)


def perspective_experiment(config, dataset_path, calib_path, device, subsample=None):
    dataset = PerspectiveDataset(dataset_path, calib_path, subsample)
    # build extractor
    extractor = expu.getObjectExtractorFromConfig(config, device)
    # get metric, sub-feature metric from config
    loc_config = config['localization']

    # extract features from all images
    scenes = {kk: extractor.detectAndComputeScene(dataset.get_image(kk))
              for kk in dataset.get_all_image_keys()}

    all_results = []
    combs = list(combinations(dataset.get_all_image_keys(), 2))
    for ii, jj in tqdm(combs):
        result = estimate_transform(dataset, loc_config, ii,
                                    scenes[ii].objects, jj, scenes[jj].objects)
        all_results.append(((ii, jj), result))
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib', help='\
Path to the dataset\'s calibration file.')
    parser.add_argument('datadir', help='\
A directory containing sub-datasets, each with images from a single DTU scene.')
    parser.add_argument('--sub', type=int, default=5,
                        help='Subsampling factor.')
    parser = expu.addCommonArgs(parser)
    args = parser.parse_args()
    device = expu.getDeviceFromArgs(args)

    all_base_cfgs = expu.getConfigsFromArgs(args)

    for datadir in Path(args.datadir).iterdir():
        if not datadir.is_dir():
            continue
        for base_cfg, cfg_path in zip(all_base_cfgs, args.cfgFiles):
            cfg_results = []
            for cfg in expu.expandConfigToParamGrid(base_cfg):
                results = perspective_experiment(cfg, datadir, args.calib, device, args.sub)
                cfg_results.append((cfg, results))
            cfg_name = Path(cfg_path).stem
            out_path = Path(datadir) / (cfg_name + '.pkl')
            with out_path.open('wb') as ff:
                pickle.dump(cfg_results, ff)


if __name__ == "__main__":
    main()
