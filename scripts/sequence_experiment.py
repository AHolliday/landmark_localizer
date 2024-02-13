import argparse
import pickle
import numpy as np
import cv2
import pykitti
import glob
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import scipy.spatial.distance as distance

from landmark_localizer import ObjectExtractor as oe
from landmark_localizer import localization as loc
from landmark_localizer import experimentUtils as expu
from landmark_localizer import constants as cs


# workaround for a bug in tqdm that raises an error
tqdm.monitor_interval = 0


ROOM_CODE_MAP = {
    'CR': 'Corridor',
    'TR': 'Terminal room',
    'RL': 'Robotics lab',
    '1PO': 'One-person office',
    '2PO': 'Two-person office',
    '2PO1': 'Two-person office',
    '2PO2': 'Two-person office',
    'CNR': 'Conference room',
    'PA': 'Printer area',
    'KT': 'Kitchen',
    'BR': 'Bathroom',
    'TL': 'Bathroom',
    'LO': 'Large office',
    'SA': 'Stairs area',
    'ST': 'Stairs area',
    'LAB': 'Lab',
}


class DummySeqExpResult:
    """Holds a collection of data that may come from multiple sources."""
    def __init__(self, data_name, exp_config, pair_results):
        self._data_name = data_name
        self.experiment_config = exp_config
        self.pair_results = pair_results

    @property
    def data_name(self):
        return self._data_name


class SequenceExperimentResult(DummySeqExpResult):
    """
    What's the relevant info we need to save?
    - dataset (KITTI, st. lucia, etc) (we're only using KITTI right now, so...)
    - sub-dataset, eg. sequence in Kitti's case
    - the configuration used to generate the objects
    - for each comparison:
        - the frames that were being compared, so that we can retrieve
        them if need be
        - the estimated transform (T, R)
        - the ground truth transform (T, R)
    """
    def __init__(self, data_seq, exp_config, pair_results):
        super().__init__(data_seq.name, exp_config, pair_results)


class MonoDatasetSequence:
    def get_all_image_paths(self):
        raise NotImplementedError()

    def get_true_pose(self, key):
        raise NotImplementedError()

    def get_image(self, image_key):
        raise NotImplementedError()


class ColdDatasetSequence(MonoDatasetSequence):
    def __init__(self, recording_dir, list_path):
        self.recording_dir = Path(recording_dir)
        self.list_path = list_path
        # self.all_image_paths = []
        with open(list_path, 'r') as ff:
            self.image_paths = sorted([Path(ll.strip()) for ll in ff])
        with (self.recording_dir / 'localization/places.lst').open('r') as ff:
            self.places = dict([ll.strip().split() for ll in ff])

    def get_image_paths(self):
        return self.image_paths
        # return [self.all_image_paths[ii] for ii in self.frames]

    def get_2d_pose(self, key):
        # strip the suffix
        stem = self.image_paths[key].stem
        parts = stem.split('_')
        contents = {}
        for part in parts:
            # the first charactar of each part indicates what it represents
            # eg. 't' for time, 'x' for x coordinate
            # the rest is a float value for that quantity
            contents[part[0]] = float(part[1:])
        return contents['x'], contents['y'], contents['a']

    def get_room_id(self, key):
        return self.places[self.image_paths[key].name]

    def get_room_type(self, key):
        room_id = self.get_room_id(key)
        room_type_code = room_id.partition('-')[0]
        return ROOM_CODE_MAP[room_type_code]

    def get_true_pose(self, key):
        """Get the ground truth info from the filename of a COLD image.

        The coordinates in the filename are 2D, of the form (x y a), where a
        is the yaw angle.  These coordinates are in a right-handed coordinate
        system where z points up.

        arguments:
        cold_filename -- the filename of the image in question
        returns:
        -- a 4x4 transform matrix giving the pose of the camera in a right-
        handed coordinate frame where y points down.
        """
        xx, yy, aa = self.get_2d_pose(key)

        # convert to a 4x4 transformation matrix in world coordinates
        mat = np.eye(4)
        mat[0, 0] = np.cos(aa)
        mat[1, 1] = mat[0, 0]
        mat[1, 0] = np.sin(aa)
        mat[0, 1] = -mat[1, 0]
        mat[0, 3] = xx
        mat[1, 3] = yy
        mat[2, 3] = 0  # let the z origin be at the camera height.

        # convert mat from x-forward y-left coords to x-right y-down coords
        mat = mat.dot(np.array([[0, 0, 1, 0],
                                [-1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]]))
        return mat

    def get_image(self, key):
        return cv2.imread(str(self.image_paths[key]))

    def get_image_path(self, key):
        return str(self.image_paths[key])

    # def plot_laser_scans(self):
    #     odom_dir = self.recording_dir / 'odom_scans'
    #     with (odom_dir / 'scans.tdf').open('r') as ff:
    #         scan_lines = [ll for ll in ff]
    #     with (odom_dir / 'odom.tdf').open('r') as ff:
    #         odom_lines = [ll for ll in ff]
    #     scans = [(float(xx) for xx in ll.split()) for ll in scan_lines]
    #     scan_idx = 0
    #     odoms = [(float(xx) for xx in ll.split()) for ll in odom_lines]
    #     for _, _, _, ts, tu, _, _, _, xx, yy, theta, _, _ in odoms:
    #         nearest_scan = None
    #         while nearest_
    #
    #
    #         ranges = scan_vals[16:]

    def __len__(self):
        return len(self.image_paths)

    @property
    def K_px(self):
        # these values taken from Ullah 2007, the technical report on COLD
        width = 640
        height = 480
        x_fov, y_fov = self.fov_rad
        x_foclen = width / (2 * np.tan(x_fov / 2))
        y_foclen = height / (2 * np.tan(y_fov / 2))
        return np.array([[x_foclen, 0, width / 2],
                         [0, y_foclen, height / 2],
                         [0, 0, 1]])

    @property
    def fov_rad(self):
        """Returns values specified in Ullah 2007, in radians"""
        return np.deg2rad([68.9, 54.4])

    @property
    def name(self):
        return ', '.join(['COLD', str(self.recording_dir),
                          str(len(self.frames)) + ' frames'])

    @property
    def image_dir(self):
        return self.recording_dir / 'std_cam'

    @property
    def kwargs(self):
        return {
            'recording_dir': str(self.recording_dir),
            'list_path': self.list_path
            }

    @property
    def frames(self):
        return [ip.stem for ip in self.image_paths]


class StereoDatasetSequence:
    """
    Should provide:
    - a list of the images in the dataset
    - a function to read in an image based on its name
    - a function to get the ground truth associated with an image
    - a function to get a list of objects associated with an image
    - est
    """

    def show_disparity_map(self, i):
        dm = self.get_disparity_map(i)
        plt.imshow(dm, 'gray')
        plt.title('right')
        plt.show()


class KittiSequence(pykitti.odometry, StereoDatasetSequence):
    def __init__(self, kitti_dir, sequence, image_names):
        """ Create a KITTI dataset object.  Based on pykitti.odometry, but
        allows element-based accessing as opposed to just iterators.

        Arguments
        kitti_dir -- the path of the directory containing the KITTI dataset.
        sequence -- the KITTI sequence (0 to 10) from which this dataset is
        drawn, as a two-digit string (eg. '07').
        image_names -- a dictionary of objects extracted from images.
        """
        frames = list(map(int, image_names))
        super().__init__(kitti_dir, sequence, frames=frames, imformat='cv2')
        self._poses_list = list(self.poses)
        T_velo_cam0 = np.linalg.inv(self.calib.T_cam0_velo)
        self.T_cam2_cam0 = self.calib.T_cam2_velo.dot(T_velo_cam0)

    def get_image_paths(self):
        # the left colour image is the default image
        return self.get_left_colour_image_paths()

    def get_image_path(self, ii):
        return str(self.get_image_paths()[ii])


    def get_left_colour_image_paths(self):
        imglob = os.path.join(self.sequence_path, 'image_2',
                              '*.{}'.format(self.imtype))
        all_paths = sorted(glob.glob(imglob))
        return [all_paths[ii] for ii in self.frames]

    def _get_colour_image(self, i, is_left):
        image_dir = 'image_'
        if is_left:
            image_dir += '2'
        else:
            image_dir += '3'
        impath = os.path.join(self.sequence_path, image_dir,
                              '*.{}'.format(self.imtype))
        imfiles = sorted(glob.glob(impath))
        return cv2.imread(imfiles[self.frames[i]])

    def _get_grayscale_image(self, i, read_code, is_left):
        image_dir = 'image_'
        if is_left:
            image_dir += '0'
        else:
            image_dir += '1'
        impath = os.path.join(self.sequence_path, image_dir,
                              '*.{}'.format(self.imtype))
        imfiles = sorted(glob.glob(impath))
        return cv2.imread(imfiles[self.frames[i]], read_code)

    def get_image(self, ii):
        # the left colour image is the default image
        return self.get_left_rgb_image(ii)

    def get_left_rgb_image(self, i):
        return self._get_colour_image(i, True)

    def get_right_rgb_image(self, i):
        return self._get_colour_image(i, False)

    def get_left_gray_image(self, i, read_code=cv2.CV_8U):
        return self._get_grayscale_image(i, read_code, True)

    def get_right_gray_image(self, i, read_code=cv2.CV_8U):
        return self._get_grayscale_image(i, read_code, False)

    def get_disparity_map(self, i):
        l_img = cv2.cvtColor(self.get_left_rgb_image(i), cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(self.get_right_rgb_image(i), cv2.COLOR_BGR2GRAY)
        return get_disparity_map(l_img, r_img)

    def get_true_pose(self, ii):
        return self.left_cam_pose(ii)

    def left_cam_pose(self, i):
        cam0_pose = self._poses_list[i]
        cam2_pose = self.T_cam2_cam0.dot(cam0_pose)
        return cam2_pose

    @property
    def K_px(self):
        return self.left_cam_K_px

    @property
    def fov_rad(self):
        """Returns x, y field-of-view in radians."""
        img_np_shape = self.get_image(0).shape
        img_shape = [img_np_shape[1], img_np_shape[0]]
        foclens = self.K_px.diagonal()[:2]
        centers = self.K_px[:2, 2]
        # compute the left- and right-hand parts of the FOV separately
        fov1 = np.arctan(centers / foclens)
        fov2 = np.arctan((img_shape - centers) / foclens)
        return fov1 + fov2

    @property
    def left_cam_K_px(self):
        return self.calib.K_cam2

    @property
    def baseline(self):
        return self.calib.b_rgb

    # the following members should be replicated in all other data-sequence
    # classes

    @property
    def name(self):
        return ', '.join(['KITTI', str(self.sequence),
                          str(len(self.frames)) + ' frames'])

    @property
    def image_dir(self):
        # return the directory of the left color images
        return Path(self.sequence_path) / 'image_2'

    @property
    def kwargs(self):
        return {
            'kitti_dir': str(Path(self.pose_path).parent),
            'sequence': self.sequence,
            'image_names': self.frames
        }


def get_disparity_map(l_img, r_img):
    """Gets the disparity map for the stereo pair, from left's POV."""
    if len(l_img.shape) > 2:
        l_img = cv2.cvtColor(l_img, cv2.CV_8U)
    if len(r_img.shape) > 2:
        r_img = cv2.cvtColor(r_img, cv2.CV_8U)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity_16bit = stereo.compute(l_img, r_img)
    # the last four bits are sub-pixel precision, so convert to float
    disparity = disparity_16bit.astype(float) / 16
    return disparity


def get_diff_between_latlons(lat1, lon1, lat2, lon2):
    """
    Like the name says.

    Computes the distance, in meters, between two lat,lon pairs, using the
    Haversine formula.  Assumes a perfectly spherical earth, so slightly
    inaccurate, but fine for our purposes.

    Arguments
    lat1, lon1, lat2, lon2 -- the coordinates, in decimal degrees.
    """
    # convert to radians
    lat1, lon1, lat2, lon2 = np.array([lat1, lon1, lat2, lon2]) * np.pi / 180
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat / 2)**2 + \
        (np.cos(lat1) * np.cos(lat2) * (np.sin(d_lon / 2)**2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return cs.earthRadiusM * c


def extract_objects(data_seq, img_idx, cached_box_dict, obj_engine):
    # get selective search parameters and extract boxes
    # WARNING: we assume here that all our configs use the same type of
    # box extraction with the same configuration.  For the present this
    # will be true, but in future it may not be.
    if isinstance(obj_engine, oe.CachedFeatureReader):
        img_path = data_seq.get_image_path(img_idx)
        objects = obj_engine.read_cached_features_as_objects(img_path)
    else:
        img = data_seq.get_image_path(img_idx)
        frame_id = data_seq.frames[img_idx]
        if not hasattr(obj_engine, 'boxFunction'):
            # just compute and return.
            objects, _ = obj_engine.detectAndComputeObjects(img)

        elif frame_id in cached_box_dict:
            # compute objects from the boxes
            boxes = cached_box_dict[frame_id]
            masks = None
            if type(boxes[0]) is tuple:
                # the list consists of (box, mask) tuples
                boxes, masks = list(zip(*boxes))
            # filtered_boxes = obj_engine.aggregateBoxes(boxes)
            objects, _ = obj_engine.getObjectsFromBoxes(img, boxes, masks)

        else:
            # compute objects and boxes, and save the boxes
            objects, _ = obj_engine.detectAndComputeObjects(img)
            cached_box_dict[frame_id] = [o.box for o in objects]

    return objects


def estimate_transform(data_seq, loc_config, from_idx, from_objs, to_idx,
                       to_objs):
    # retrieve ground truth transform
    gt_Mp_w1 = data_seq.get_true_pose(from_idx)
    gt_Mp_w2 = data_seq.get_true_pose(to_idx)

    gt_Mp_2w = np.linalg.inv(gt_Mp_w2)
    gt_Mp_21 = gt_Mp_2w.dot(gt_Mp_w1)
    gt_Rp_21 = gt_Mp_21[:3, :3]
    gt_Tp_21 = gt_Mp_21[:3, 3]

    # save the results
    pair_result = {
        'gt_T21': gt_Tp_21,
        'gt_R21': gt_Rp_21
        }

    # estimate the transform
    foclen_px = (data_seq.K_px[0, 0] +
                 data_seq.K_px[1, 1]) / 2
    pp_px = (data_seq.K_px[0, 2], data_seq.K_px[1, 2])
    loc_args = (from_objs, to_objs, foclen_px, pp_px)
    try:
        tmp_ret = loc.findCameraTransformFromObjects(*loc_args, **loc_config)
    except loc.LocalizationFailure as lf:
        # print(lf)
        return pair_result

    E, Rp_21s, Tp_21s, N1s, pts3ds, matchSets, pts1, pts2 = tmp_ret

    # compute stereo disparity and use it to scale 3d points
    Tp_21 = Tp_21s[0]
    pts3d = pts3ds[0]

    if isinstance(data_seq, StereoDatasetSequence):
        scales = []
        # data_seq.show_disparity_map(from_idx)
        disparity_map = data_seq.get_disparity_map(from_idx)
        for pt3d, pt2d in zip(pts3d, pts1):
            # get the disparity at that coordinate in the map
            ipt2d = np.rint(pt2d).astype(int)
            pt_disp = disparity_map[ipt2d[1], ipt2d[0]]
            if pt_disp > 0:
                # negative disparities could not be computed, ignore them.
                # disparity(x) = (x - x') = B * f / Z_abs
                # -> Z_abs = B * f / disparity(x)
                # scale factor = Z_abs / Z_rel
                # -> scale factor = B * f / (disparity(x) * Z_rel)
                scale = data_seq.baseline * foclen_px / (pt3d[2] * pt_disp)
                scales.append(scale)
        if len(scales) == 0:
            return pair_result
        Tp_21 *= np.median(scales)
    elif isinstance(data_seq, ColdDatasetSequence):
        # TODO scale based on laser odometry?
        pass

    pair_result['est_T21'] = Tp_21.T
    pair_result['est_R21'] = Rp_21s[0]
    return pair_result


def localize_over_gaps(data_seq, config, max_gap, subsample, clear,
                       cache_filename=None, device=None):
    """
    Localize across gaps of up to max-gap size.
    """
    obj_engine = expu.getObjectExtractorFromConfig(config, device)

    # read in selective search boxes
    cached_box_dict = {}
    cache_path = None
    if cache_filename:
        cache_path = data_seq.image_dir / cache_filename
        if not clear and cache_path.is_file():
            # read in the boxes
            with cache_path.open('rb') as f:
                cached_box_dict = pickle.load(f)

    pair_results = []
    loc_config = config['localization']

    all_idxs = list(range(len(data_seq)))
    if subsample is None:
        idx_sets = [all_idxs]
    else:
        idx_sets = [all_idxs[j::subsample] for j in range(subsample)]

    pbar = tqdm(total=sum([len(idxs) for idxs in idx_sets]))
    for idxs in idx_sets:
        # initialize the object-list queue
        obj_queue = []
        for gg in range(max_gap):
            objects = extract_objects(data_seq, idxs[gg], cached_box_dict,
                                      obj_engine)
            obj_queue.append((idxs[gg], objects))

        for ii in range(len(idxs)):
            # update the queue
            i, from_objs = obj_queue.pop(0)
            if max_gap < len(idxs) - ii:
                j = idxs[ii + max_gap]
                gap_end_objs = extract_objects(data_seq, j, cached_box_dict,
                                               obj_engine)
                obj_queue.append((j, gap_end_objs))

            # if len(from_objs) == 0:
            #     continue
            #     pbar.update(1)

            for j, to_objs in obj_queue:
                if len(from_objs) == 0 or len(to_objs) == 0:
                    continue
                pair_key = (i, j)
                result = estimate_transform(data_seq, loc_config, i, from_objs,
                                            j, to_objs)
                pair_results.append((pair_key, result))

            pbar.update(1)
    pbar.close()

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_box_dict, f)
    return pair_results


def kitti_seq_from_list(root_data_dir, list_path):
    with open(list_path, 'r') as ff:
        lines = ff.readlines()
    frames = [os.path.splitext(os.path.basename(ll.strip()))[0]
              for ll in lines]

    # set up the data sequence
    img_path = lines[0].strip()
    path_parts = img_path.split(os.sep)
    sequences_dir_idx = path_parts.index('sequences')
    seq_str = path_parts[sequences_dir_idx + 1]
    return KittiSequence(root_data_dir, seq_str, frames)


def get_arg_parser(parser=None):
    """Set up arguments for running this script directly."""
    if parser is None:
        parser = argparse.ArgumentParser()
    expu.addCommonArgs(parser)
    parser.add_argument('--list', action='append', required=True, help='\
The text list describing a sequence of images to experiment on.')
    parser.add_argument('--cold', action='store_true',
                        help='If provided, use the COLD dataset.')
    parser.add_argument('--data', help='Root directory of the dataset.')
    parser.add_argument('-n', type=int, default=cs.defaultMaxKittiSpan, help='\
The largest gap between frames to compare.')
    parser.add_argument('--clear', action='store_true', help='\
recompute boxes instead of reading the saved ones.')
    parser.add_argument('--reuse', help='\
Reuse pre-computed bounding boxes, and save new bounding boxes for reuse.')
#     parser.add_argument('--start', type=int, default=0, help='\
# Subsample the data list by the specified factor.')
    parser.add_argument('--sub', type=int, help='\
Subsample the data list by the specified factor.')
#     parser.add_argument('--stop', type=int, help='\
# Subsample the data list by the specified factor.')
    return parser


def main(args):
    """Run sequence experiments with the provided configs and sequence."""
    # read in the pickle file
    # set up root directory
    root_data_dir = args.data
    device = expu.getDeviceFromArgs(args)
    if not root_data_dir:
        if args.cold:
            root_data_dir = '/home/a.holliday/COLD/saarbrucken/seq2_cloudy1'
        else:
            root_data_dir = cs.defaultKittiDir

    for list_path in args.list:
        if args.cold:
            data_seq = ColdDatasetSequence(root_data_dir, list_path)
        else:
            data_seq = kitti_seq_from_list(root_data_dir, list_path)

        all_base_cfgs = expu.getConfigsFromArgs(args)
        for base_cfg, cfg_path in zip(all_base_cfgs, args.cfgFiles):
            results = []
            for cfg in expu.expandConfigToParamGrid(base_cfg):
                # run the experiment
                print('Now running', cfg['name'])
                cache_filename = None
                if args.reuse:
                    cache_filename = args.reuse + '.pkl'
                cfg_results = localize_over_gaps(data_seq, cfg, args.n,
                                                 args.sub, args.clear,
                                                 cache_filename, device)
                results.append(SequenceExperimentResult(data_seq, cfg,
                                                        cfg_results))

            # base output filename on input list filename
            list_dir, list_filename = os.path.split(list_path)
            list_name = os.path.splitext(list_filename)[0]
            cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
            out_filename = '_'.join([cfg_name, 'results', list_name, 'gap',
                                     str(args.n) + '.pkl'])
            out_path = os.path.join(list_dir, out_filename)
            with open(out_path, 'wb') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
