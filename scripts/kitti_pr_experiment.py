import argparse
import h5py
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import combinations, product
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import tempfile
import yaml
import subprocess
import datetime
import cv2

from sequence_experiment import KittiSequence, ColdDatasetSequence, \
    estimate_transform, kitti_seq_from_list

from landmark_localizer import constants as cs
from landmark_localizer import experimentUtils as expu
from landmark_localizer import torchUtils as tchu
from landmark_localizer import localization as loc
from landmark_localizer import geometryUtils as geo
from landmark_localizer.ObjectExtractor import Scene


MATCH_GRP_KEY = 'object matches'
SIM_KEY = 'similarities'
CNST_KEY = 'consistencies'


KITTI_DISTS = [2, 5, 10, 20, 40, 80]
COLD_DISTS = [0.1, 2, 4, 6, 8, 10]


class MultiDistPrResultsContainer:
    def __init__(self, data_seq, config, sim_matrix):
        self.data_seq_kwargs = data_seq.kwargs
        self.data_seq_type = type(data_seq)
        self.config = config
        self.sim_matrix = sim_matrix
        self.dist_experiments = defaultdict(list)

    def add_dist_experiment(self, step_dist_m, dist_exp_result):
        self.dist_experiments[step_dist_m].append(dist_exp_result)

    @property
    def data_seq(self):
        if self.data_seq_type == KittiSequence and \
           not Path(self.data_seq_kwargs['kitti_dir']).exists():
            # this may have been computed on a different machine.  Change it
            # to point to the default kitti directory on this machine.
            self.data_seq_kwargs['kitti_dir'] = cs.defaultKittiDir
        elif self.data_seq_type == ColdDatasetSequence and \
           not Path(self.data_seq_kwargs['recording_dir']).exists():
            self.data_seq_kwargs['recording_dir'] = cs.defaultColdDir
            self.data_seq_kwargs['list_path'] = '../saarbrucken/seq2_cloudy1/sub2.txt'
        return self.data_seq_type(**self.data_seq_kwargs)


class FabmapResultsContainer(MultiDistPrResultsContainer):
    def __init__(self, data_seq, config, fabmap_config):
        super().__init__(data_seq, config, None)
        self.fabmap_config = fabmap_config


class PrLocResult:
    """What's the relevant info we need to save?
    - dataset (KITTI, st. lucia, etc) (we're only using KITTI right now, so...)
    - sub-dataset, eg. sequence in Kitti's case
    - the minimum distance used to separate frames
    - the set of sub-indexes
    - the configuration used to generate the objects
    - the set of matches found
    - for each match found:
        - the estimated pose
        - the ground truth pose

    """
    def __init__(self, matches, est_poses):
        self.matches = matches
        self.est_poses = est_poses

    @property
    def sub_idxs(self):
        return [qi for qi, _ in self.matches]


def dbkey_from_index(index):
    return 'frame ' + str(index)


def dbkey_from_index_pair(ii, jj):
    return str(ii) + ', ' + str(jj)


def compute_transform_matrices(data_seq):
    idxs = list(range(len(data_seq)))
    poses = np.array([data_seq.get_true_pose(ii) for ii in idxs])
    pos_dists = np.zeros((len(idxs), len(idxs)))
    rot_dists = np.zeros((len(idxs), len(idxs)))
    for ii in idxs:
        for jj in idxs:
            pos_dist = np.linalg.norm(poses[ii][:3, 3] - poses[jj][:3, 3])
            pos_dists[ii, jj] = pos_dist
            rot_dists[ii, jj] = geo.rotationMatrixDistance(poses[ii][:3, :3],
                                                           poses[jj][:3, :3])
    # remove negative values that occur due to rounding error
    rot_dists[rot_dists < 0] = 0
    return pos_dists, rot_dists


def plot_dist_histogram(pos_dist_matrix):
    consecutive_dists = pos_dist_matrix[range(len(pos_dist_matrix) - 1),
                                        range(1, len(pos_dist_matrix))]
    print('distance mean:', consecutive_dists.mean(), 'median:',
          np.median(consecutive_dists))
    plt.hist(consecutive_dists, bins=range(11))
    plt.show()


def extract_scenes(data_seq, config, db_path, device):
    obj_engine = expu.getObjectExtractorFromConfig(config, device)
    scenes = []
    with h5py.File(db_path, 'a') as object_db:
        for ii in tqdm(range(len(data_seq))):
            frame_key = dbkey_from_index(ii)
            if frame_key in object_db:
                scene = Scene.fromHdf5(object_db, frame_key)
            else:
                # extract the scene and add it to the object_db
                img = data_seq.get_image(ii)
                scene = obj_engine.detectAndComputeScene(img)
                scene.toHdf5(object_db, frame_key)
            scenes.append(scene)
            object_db.flush()
    return scenes


def compute_similarities(scenes, db_path, device, metric, all_to_device=True):
    """all_to_device: if True, put all the features on the device at once.
        If False, move only one set of image features onto the device at a
        time, to save device memory.
    """
    # set up the imageXimage matrix
    sim_matrix = np.zeros((len(scenes), len(scenes)))
    with h5py.File(db_path, 'r') as object_db:
        if 'similarities' in object_db:
            return

    # compute the similarity scores
    all_features = [sc.descriptors for sc in scenes]
    if all_to_device:
        # move all the features to the specified device at once
        all_features = [torch.tensor(dd, device=device) for dd in all_features]

    pbar = tqdm(total=len(scenes) * (len(scenes) - 1) // 2)
    for ii, featsi in enumerate(all_features):
        if not all_to_device:
            featsi = torch.tensor(featsi).to(device)
        scenei = scenes[ii]
        for jj, featsj in enumerate(all_features[ii + 1:], ii + 1):
            if len(featsi) == 0 or len(featsj) == 0:
                # can't say anything about an empty scene
                score = 0
                matches = np.array([])
            else:
                if not all_to_device:
                    featsj = torch.tensor(featsj).to(device)
                scenej = scenes[jj]
                # do matching and compute the similarity score
                matches, dists = tchu.matchByBruteForce(featsi, featsj, metric)
                shape_scores = [loc.computeAspectRatioScore(scenei.boxes[mi],
                                                            scenej.boxes[mj])
                                for mi, mj in matches]
                shape_scores = np.array(shape_scores)
                score = (len(dists) - (shape_scores * dists).sum())
                normalizer = np.sqrt(len(featsi) * len(featsj))
                score /= normalizer
            sim_matrix[ii, jj] = score
            pbar.update(1)
    pbar.close()
    # copy the above-diagonal elements below the diagonal.
    sim_matrix += sim_matrix.T

    # store the similarity matrix in the database.
    with h5py.File(db_path, 'a') as object_db:
        if SIM_KEY in object_db:
            object_db[SIM_KEY][...] = sim_matrix
        else:
            object_db[SIM_KEY] = sim_matrix
        object_db.flush()


def build_idx_sequences(data_seq, step_dists, scenes=None,
                        num_starts_per_dist=None):
    pos_dist_matrix, rot_dist_matrix = compute_transform_matrices(data_seq)
    sub_idx_sets = []
    for step_dist_m in step_dists:
        # assemble the list of different indices to start from
        start_idxs = [0]
        ii = 1
        if num_starts_per_dist:
            start_threshold = step_dist_m / num_starts_per_dist
        else:
            # just use the smallest step distance
            start_threshold = step_dists[0]
        while pos_dist_matrix[0, ii] < step_dist_m:
            if pos_dist_matrix[start_idxs[-1], ii] >= start_threshold:
                start_idxs.append(ii)
            ii += 1

        # given a starting index, collect other indices with the given degree
        # of distance between them.

        # use the horizontal FOV to limit turns
        angle_limit = data_seq.fov_rad[0]
        for start_idx in start_idxs:
            sub_idxs = [start_idx]
            ii = start_idx
            jj = ii + 1
            while jj < len(data_seq):
                has_features = scenes is None or len(scenes[jj]) > 0
                if has_features and (
                    pos_dist_matrix[ii, jj] >= step_dist_m or
                    np.arccos(1 - rot_dist_matrix[ii, jj]) >= angle_limit):
                        sub_idxs.append(jj)
                        ii = jj

                    # # the below method avoids placing multiple views at
                    # # scenes that have been visited more than once
                    # j_is_near = pos_dist_matrix[jj][sub_idxs] < step_dist_m
                    # angles = np.arccos(1 - rot_dist_matrix[jj][sub_idxs])
                    # j_faces_same_dir = angles < angle_limit
                    # if not np.any(j_is_near & j_faces_same_dir):
                    #     sub_idxs.append(jj)
                    #     ii = jj
                jj += 1
            sub_idx_sets.append((step_dist_m, sub_idxs))
    return sub_idx_sets


def pr_experiment(data_seq, sub_idxs, scenes, sim_matrix, loc_cfg, pbar=None,
                  check_consistency=True):
    if not sub_idxs:
        sim_matrix = sim_matrix.copy()
        sub_idxs = tuple(range(len(scenes)))
    else:
        # zero out all locations that weren't selected by sub_idxs
        sim_matrix_tmp = np.zeros(sim_matrix.shape)
        sub_is, sub_js = list(zip(*product(sub_idxs, sub_idxs)))
        sim_matrix_tmp[sub_is, sub_js] = sim_matrix[sub_is, sub_js]
        sim_matrix = sim_matrix_tmp

    # ignore a scene's similarity with itself.
    np.fill_diagonal(sim_matrix, 0)

    query_matches = []
    est_poses = []
    def add_match(i1, i2, loc_result=None):
        """Helper function to avoid duplicated code"""
        query_matches.append((i1, i2))
        est_Tp_21 = None
        if loc_result is not None and 'est_T21' in loc_result:
            est_Tp_21 = geo.buildTfMatrix(R=loc_result['est_R21'],
                                          T=loc_result['est_T21'])
        est_poses.append(est_Tp_21)

    all_num_failed = np.zeros(sim_matrix.shape[0])
    for qi in sub_idxs:
        candidates = sim_matrix[qi]
        while np.any(candidates > 0):
            ci = candidates.argmax()
            if not check_consistency or (len(scenes[qi]) == 0 and
                                         len(scenes[ci]) == 0):
                # accept the match without a localization result
                add_match(qi, ci)
                break
            if check_consistency and (len(scenes[qi]) > 0 and
                                      len(scenes[ci]) > 0):
                loc_result = estimate_transform(data_seq, loc_cfg, qi,
                                                scenes[qi].objects, ci,
                                                scenes[ci].objects)
                add_match(qi, ci, loc_result)
                break
            # if we made it this far, we tried for consistency and failed
            all_num_failed[qi] += 1
            candidates[ci] = 0
            sim_matrix[ci, qi] = 0

        if np.all(candidates == 0):
            # we couldn't find any consistent candidate.
            add_match(qi, None)

        if pbar:
            # update the progress bar if one was provided
            pbar.update(1)
    if np.any(all_num_failed > 0):
        print('average num. failures:', all_num_failed.mean())
    return PrLocResult(query_matches, est_poses)


def run_all_step_experiments(data_seq, scenes, db_path, cfg, step_dists,
                             check_consistency):
    loc_cfg = cfg['localization']
    with h5py.File(db_path, 'r') as object_db:
        sim_matrix = object_db[SIM_KEY][...]

    all_results = MultiDistPrResultsContainer(data_seq, cfg, sim_matrix)
    sub_idx_sets = build_idx_sequences(data_seq, step_dists)
    pbar = tqdm(total=sum(len(si) for _, si in sub_idx_sets))
    for step_dist_m, sub_idxs in sub_idx_sets:
        pr_result = pr_experiment(data_seq, sub_idxs, scenes, sim_matrix,
                                  loc_cfg, pbar, check_consistency)
        # print('num result matches:', len(pr_result.matches))
        all_results.add_dist_experiment(step_dist_m, pr_result)
    pbar.close()
    return all_results


def run_all_fabmap_experiments(data_seq, scenes, step_dists, cfg, fabmap_cfg,
                               fabmap_cli_path, check_consistency,
                               use_scene_features=False):
    # for each sub index list:
    suffix = '.yml' if use_scene_features else '.txt'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tmp_list_path = tf.name
    paths_cfg = fabmap_cfg['FilePaths']
    paths_cfg['TestPath'] = tmp_list_path
    if use_scene_features:
        fabmap_cfg['FeaturesFromFile'] = 1.
    else:
        fabmap_cfg['FeaturesFromFile'] = 0.

    # generate an appropriate openfabmap config
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yml') as cf:
        fabmap_cfg_path = cf.name

    all_img_paths = data_seq.get_image_paths()
    all_results = FabmapResultsContainer(data_seq, cfg, fabmap_cfg)
    # generate sub index lists
    sub_idx_sets = build_idx_sequences(data_seq, step_dists)
    print('subsequences generated')

    pbar = tqdm(total=sum(len(si) for _, si in sub_idx_sets))
    for step_dist, sub_idxs in sub_idx_sets:
        # remove any intermediate openfabmap files
        if Path(tmp_list_path).exists():
            Path(tmp_list_path).unlink()
        if Path(paths_cfg['TestImageDesc']).exists():
            Path(paths_cfg['TestImageDesc']).unlink()
        if Path(paths_cfg['FabMapResults']).exists():
            Path(paths_cfg['FabMapResults']).unlink()
        if Path(paths_cfg['FeatureFailures']).exists():
            Path(paths_cfg['FeatureFailures']).unlink()

        # generate an image path list file for these indexes
        sub_paths = [all_img_paths[ii] for ii in sub_idxs]

        # write features to tmp_list_path (change it to YAML)
        if use_scene_features:
            fs = cv2.FileStorage(tmp_list_path, cv2.FILE_STORAGE_WRITE)
            for idx in sub_idxs:
                fs.write('img' + str(idx), scenes[idx]._descriptors)
            fs.release()
        else:
            with open(tmp_list_path, 'w') as ff:
                ff.writelines([str(sp) + '\n' for sp in sub_paths])

        # extract features and run fabmap
        for func_str in ['GenerateFABMAPTestData', 'RunOpenFABMAP1by1']:
            fabmap_cfg['Function'] = func_str
            with open(fabmap_cfg_path, 'w') as ff:
                yaml.dump(fabmap_cfg, ff, version=(1, 0))
            retcode = subprocess.call([fabmap_cli_path, '-s', fabmap_cfg_path])
            if retcode is not 0:
                raise Exception('retcode ' + str(retcode) + ' recieved.')

        # read in the resulting confusion matrix and expand it to full size
        sub_conf_mat = np.loadtxt(paths_cfg['FabMapResults'])
        conf_mat = np.zeros((len(data_seq), len(data_seq)))
        # skip any indexes for which no features could be extracted
        if Path(paths_cfg['FeatureFailures']).exists():
            failure_list = np.loadtxt(paths_cfg['FeatureFailures']).astype(int)
            # remove any sub_idxs for which no features could be detected
            sub_idxs = [si for ii, si in enumerate(sub_idxs)
                        if ii not in failure_list]
        sub_is, sub_js = list(zip(*product(sub_idxs, sub_idxs)))
        conf_mat[sub_is, sub_js] = sub_conf_mat.flatten()
        conf_mat_nonew = conf_mat.copy()
        np.fill_diagonal(conf_mat_nonew, 0)
        pr_result = pr_experiment(data_seq, sub_idxs, scenes,
                                  conf_mat_nonew, cfg['localization'], pbar)
        pr_result.conf_matrix = conf_mat
        all_results.add_dist_experiment(step_dist, pr_result)
        # pbar.update(len(sub_idxs))
    pbar.close()
    return all_results


def plot_map_viewpoints(data_seq, step_dists, scenes):
    sub_idx_sets = build_idx_sequences(data_seq, step_dists, scenes)
    all_xs, all_ys = [], []
    for idx in range(len(data_seq)):
        xx, yy, aa = data_seq.get_2d_pose(idx)
        all_xs.append(xx)
        all_ys.append(yy)
    for ii, (dist, sub_idxs) in enumerate(sub_idx_sets):
        xs = []
        ys = []
        angles = []
        for idx in sub_idxs:
            xx, yy, aa = data_seq.get_2d_pose(idx)
            xs.append(xx)
            ys.append(yy)
            angles.append(aa)
        heads = [(np.cos(aa), np.sin(aa)) for aa in angles]
        dxs, dys = zip(*heads)
        plt.plot(all_xs, all_ys, label='full')
        plt.scatter(xs, ys, label=str(dist) + ' m, ' + str(ii))
        for xx, yy, dx, dy in zip(xs, ys, dxs, dys):
            plt.arrow(xx, yy, dx, dy, head_width=0.3, head_length=0.3,
                      fc='k', ec='k')
        plt.legend()
        filename = '_'.join(['map', str(dist), 'm', str(ii)]) + '.jpg'
        filepath = Path('/home/a.holliday/landmark_localizer/') / filename
        plt.savefig(filepath)
        plt.figure()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='append', required=True, help='\
The text list describing a sequence of images to experiment on.')
    parser.add_argument('--data', dest='data_dir',
                        help='Root directory of the dataset.')
    parser.add_argument('--cold', action='store_true',
                        help='If provided, use the COLD dataset.')
    parser.add_argument('--db', help='\
Path to the hdf5 database in which to store computed features and other data.')
    parser.add_argument('--pkl', help='\
Path to the pickle file in which to store the results.')
    parser.add_argument('--ff', action='store_true',
                        help='Compute the fabmap features ourselves.')
    parser.add_argument('--nocc', action='store_true',
                        help='Do not perform a spatial consistency check.')

    exp_type_grp = parser.add_mutually_exclusive_group()
    exp_type_grp.add_argument('--fabmap', help='\
    The root fabmap directory.  If provided, run fabmap experiments.')
    exp_type_grp.add_argument('--distmat', help='\
    Path to a distance matrix .csv file.  If provided, use the distances \
contained in this file to answer queries.')

    parser = expu.addCommonArgs(parser)
    args = parser.parse_args()
    device = expu.getDeviceFromArgs(args)

    if args.cold:
        step_dists_m = COLD_DISTS
    else:
        step_dists_m = KITTI_DISTS

    all_results = []

    check_consistency = not args.nocc
    root_data_dir = args.data_dir
    if not root_data_dir:
        if args.cold:
            root_data_dir = cs.defaultColdDir
        else:
            root_data_dir = cs.defaultKittiDir

    if args.fabmap:
        fm_path = Path(args.fabmap)
        fm_cli_path = fm_path / 'build/bin/openFABMAPcli'
        with (fm_path / 'samples/pr_settings.yml').open('r') as ff:
            fm_cfg = yaml.load(ff)

    for list_path in args.list:
        print('running', list_path)
        if args.cold:
            data_seq = ColdDatasetSequence(root_data_dir, list_path)
        else:
            data_seq = kitti_seq_from_list(root_data_dir, list_path)

        # plot_map_viewpoints(data_seq, step_dists_m, None)
        # exit()
        # pm, _ = compute_transform_matrices(data_seq)
        # plot_dist_histogram(pm)
        # exit()

        list_name = Path(list_path).stem
        for base_cfg in expu.getConfigsFromArgs(args):
            for cfg in expu.expandConfigToParamGrid(base_cfg):
                if args.db:
                    db_path = args.db
                else:
                    db_path = list_name + '_' + cfg['name'] + '.hdf5'
                    if args.cold:
                        db_path = 'cold_' + db_path

                scenes = extract_scenes(data_seq, cfg, db_path, device)
                if args.fabmap:
                    result = run_all_fabmap_experiments(
                        data_seq, scenes, step_dists_m, cfg, fm_cfg,
                        fm_cli_path, check_consistency, args.ff)

                elif args.distmat:
                    distmat = np.genfromtxt(args.distmat, delimiter=",")
                    # extra columns may exist
                    dimlen = min(distmat.shape)
                    distmat = distmat[0:dimlen, 0:dimlen]
                    # similarities are negative distances
                    simmat = distmat.max() - distmat
                    with h5py.File(db_path, 'a') as db:
                        if SIM_KEY in db:
                            del db[SIM_KEY]
                        db[SIM_KEY] = simmat

                    result = run_all_step_experiments(
                        data_seq, scenes, db_path, cfg, step_dists_m,
                        check_consistency)

                else:
                    metric = cfg['localization']['metric']
                    compute_similarities(scenes, db_path, device, metric)
                    result = run_all_step_experiments(
                        data_seq, scenes, db_path, cfg, step_dists_m,
                        check_consistency)

                all_results.append(result)

    if args.pkl:
        pickle_path = args.pkl
    else:
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pickle_path = 'kitti_pr_' + datetime_str + '.pkl'
        if args.fabmap:
            pickle_path = 'fabmap_' + pickle_path
        elif args.distmat:
            dmpath = Path(args.distmat)
            pickle_path = str(dmpath.parent / dmpath.stem) + '_' + pickle_path
        if args.cold:
            pickle_path = 'cold_' + pickle_path
    with open(pickle_path, 'wb') as ff:
        pickle.dump(all_results, ff)


if __name__ == "__main__":
    main()
