import argparse
import pickle
import numpy as np
from itertools import combinations
from tqdm import tqdm
import copy
import torch

from realWorldExperiment import extractScenesFromDataset
from landmark_localizer import localization as loc
from landmark_localizer import experimentUtils as expu
from landmark_localizer import constants as consts
from landmark_localizer import torchUtils as tchu

# workaround for a bug in tqdm that raises an error
tqdm.monitor_interval = 0


class MontrealPrResult:
    def __init__(self, sim_matrix, image_paths, config):
        # TODO also store object matches and distances?
        self.sim_matrix = sim_matrix
        self.image_paths = image_paths
        self.config = config

    @property
    def matches(self):
        return np.argmax(self.sim_matrix, axis=1)


def matched_objs_are_consistent(matched_objs1, matched_objs2, loc_kwargs,
                                sub_feat_metric='euclidean'):
    pts1, pts2 = [], []
    have_subs = any([len(ob.getSubFeatures()) > 0
                     for ob in matched_objs1 + matched_objs2])
    if have_subs and sub_feat_metric:
        matched_sub_pts, dists = loc.matchSubFeatures(matched_objs1,
                                                      matched_objs2,
                                                      sub_feat_metric,
                                                      crossCheck=True)
        if len(matched_sub_pts) > 0:
            points1, points2 = [list(_) for _ in zip(*matched_sub_pts)]
    if len(pts1) < 5:
        points1 += [ob.center for ob in matched_objs1]
        points2 += [ob.center for ob in matched_objs2]

    # the specific focal length and principal point aren't important for
    # determining whether a valid transform can be found at all.
    focal = 1
    pp = (0.5, 0.5)
    try:
        if loc_kwargs['planar']:
            loc.findTransformFrom2dMatches(points1, points2, focal, pp,
                                           loc_kwargs['ransacThreshold'])
        else:
            loc.findTransformFrom3dMatches(points1, points2, focal, pp,
                                           loc_kwargs['ransacThreshold'])
        return True
    except loc.LocalizationFailure:
        return False


def montreal_pr_experiment(cfg, reuse, hardmode):
    img_paths = []
    scenes = []
    loc_kwargs = cfg['localization']
    obj_engine = expu.getObjectExtractorFromConfig(cfg)
    for ds in tqdm(consts.outdoorDatasets):
        if hardmode:
            img_paths.append(ds.images[0])
            img_paths.append(ds.images[-1])
        else:
            img_paths += ds.images
        scenes += extractScenesFromDataset(obj_engine, ds, reuse)

    all_imgs_tensors = [torch.tensor(scene.descriptors).to(obj_engine.device)
                        for scene in scenes]
    pbar = tqdm(total=len(img_paths) * (len(img_paths) - 1) // 2)
    sim_matrix = np.zeros((len(img_paths), ) * 2)
    img_to_img_matches = {}

    # compute similarity matrix between all image pairs, based only on objects
    for i1 in range(len(img_paths)):
        feats1 = all_imgs_tensors[i1]
        for i2 in range(i1 + 1, len(img_paths)):
            feats2 = all_imgs_tensors[i2]
            matches, dists = tchu.matchByBruteForce(feats1, feats2)
            shape_scores = [loc.computeAspectRatioScore(scenes[i1].boxes[m1],
                                                        scenes[i2].boxes[m2])
                            for m1, m2 in matches]
            shape_scores = np.array(shape_scores)
            normalizer = np.sqrt(len(feats1) * len(feats2))
            score = (len(dists) - (shape_scores * dists).sum()) / normalizer
            sim_matrix[i1, i2] = score
            img_to_img_matches[(i1, i2)] = matches
            img_to_img_matches[(i2, i1)] = np.flip(matches, axis=1)
            pbar.update(1)

    # copy the above-diagonal entries of sim_matrix to below the diagonal
    sim_matrix += sim_matrix.T

    # find the highest-scoring spatially-consistent candidate.  This is an
    # optimization: this saves us from having to match sub-features and check
    # consistency on every pair of images.  We only do it on promising
    # candidates.  Gives us a speedup of ~10% with no quality cost.
    all_num_failed = np.zeros(sim_matrix.shape[0])
    for qi in tqdm(range(sim_matrix.shape[0])):
        candidates = sim_matrix[qi]
        while np.any(candidates > 0):
            ci = candidates.argmax()
            matches = img_to_img_matches[(qi, ci)]
            qobjs = scenes[qi].objects
            cobjs = scenes[ci].objects
            mobjs1, mobjs2 = list(zip(*[(qobjs[m1], cobjs[m2])
                                        for m1, m2 in matches]))
            if matched_objs_are_consistent(mobjs1, mobjs2, loc_kwargs):
                break
            all_num_failed[qi] += 1
            candidates[ci] = 0
            sim_matrix[ci, qi] = 0
    print('average num. failures:', all_num_failed.mean())

    result = MontrealPrResult(sim_matrix, img_paths, cfg)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle', help='Name of output pickle file.')
    parser.add_argument('--reuse', help='\
Reuse pre-computed bounding boxes, and save new bounding boxes for reuse.')
    parser.add_argument('--hardmode', action='store_true', help='\
Only include maximally-separated images from scenes.')
    parser = expu.addCommonArgs(parser)
    args = parser.parse_args()

    exp_results = []
    for base_cfg in expu.getConfigsFromArgs(args):
        for cfg in expu.expandConfigToParamGrid(base_cfg):
            exp_result = montreal_pr_experiment(cfg, args.reuse, args.hardmode)
            exp_results.append(exp_result)

    with open(args.pickle, 'wb') as ff:
        pickle.dump(exp_results, ff)
