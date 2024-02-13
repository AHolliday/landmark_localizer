#!/usr/bin/env python
import cv2
import os
import argparse
import pickle
import yaml
import numpy as np
from tqdm import tqdm
from itertools import product
from pathlib import Path

from landmark_localizer import experimentUtils as expu
from landmark_localizer import geometryUtils as geo
from landmark_localizer import myCvTools as mycv
from landmark_localizer import localization as loc
from landmark_localizer import constants as consts
from landmark_localizer.ObjectExtractor import CachedFeatureReader

np.set_printoptions(precision=3)


TYPE_HOMOG = 'Homography'
TYPE_FUND = 'Fundamental matrix'
TYPE_FAILURE = 'Localization Failure'


class Result(object):
    @classmethod
    def FailureResult(cls, dataset, far_image_idx, near_image_idx, img_shape):
        result = cls(dataset, far_image_idx, near_image_idx, img_shape,
                     None, None, None, None, result_type=TYPE_FAILURE)
        return result

    def __init__(self, dataset, far_image_idx, near_image_idx, img_shape,
                 far_pts, near_pts, M, inlier_matches, result_type=TYPE_HOMOG):
        self.result_type = result_type
        self._dataset = dataset
        self.far_image_idx = far_image_idx
        self.near_image_idx = near_image_idx
        self.far_pts = np.array(far_pts)
        self.near_pts = np.array(near_pts)
        self.matrix = M
        self.inlier_matches = np.array(inlier_matches)
        self.img_shape = img_shape

    @property
    def dataset(self):
        if not hasattr(self, '_dataset'):
            # rename the old member
            self._dataset = self.__dict__['dataset']

        if not Path(self._dataset.dir).exists():
            new_dir = Path(consts.mtl_dir) / Path(self._dataset.dir).stem
            self._dataset.dir = new_dir
            
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset):
        self._dataset = new_dataset

    @property
    def is_failure(self):
        return self.result_type == TYPE_FAILURE

    @property
    def is_homography(self):
        return self.result_type == TYPE_HOMOG

    @property
    def is_fundamental(self):
        return self.result_type == TYPE_FUND

    @property
    def H(self):
        if self.is_homography:
            return self.matrix
        else:
            return None

    @property
    def F(self):
        if self.is_fundamental:
            return self.matrix
        else:
            return None

    @property
    def far_inlier_pts(self):
        return self.far_pts[self.inlier_matches[:, 0]]

    @property
    def near_inlier_pts(self):
        return self.near_pts[self.inlier_matches[:, 1]]

    @property
    def data_pair_name(self):
        format_str = '{} {} vs {}'
        return format_str.format(os.path.basename(self.dataset.dir),
                                 self.far_image_idx, self.near_image_idx)

    @property
    def save_friendly_data_pair_name(self):
        return '{}{}_{}'.format(self.far_image_idx, self.near_image_idx,
                                os.path.basename(self.dataset.dir))

    @property
    def images(self):
        far_img_filename = '{}.png'.format(self.far_image_idx)
        far_image = cv2.imread(os.path.join(self.dataset.dir,
                                            far_img_filename))
        near_img_filename = '{}.png'.format(self.near_image_idx)
        near_image = cv2.imread(os.path.join(self.dataset.dir,
                                             near_img_filename))
        return far_image, near_image

    def score(self, threshold):
        if self.is_failure:
            return 0
        gt_errors = self._separate_gt_errors()
        far_gt_pts, near_gt_pts = self.get_ground_truth()
        gt_scores = threshold - np.array(gt_errors)
        return sum([s for s in gt_scores if s > 0])

    def _separate_gt_errors(self):
        if self.is_failure:
            return None
        far_gt_pts, near_gt_pts = self.get_ground_truth()
        gt_pairs = list(zip(far_gt_pts, near_gt_pts))
        # compute errors of each ground-truth match individually
        if self.is_homography:
            error_func = symmetricTransferError
        else:
            error_func = fundMatReprojectionError
        pt_errors = [error_func(self.matrix, f[np.newaxis], n[np.newaxis])
                     for f, n in gt_pairs]
        return pt_errors

    def error(self):
        if self.is_failure:
            return None
        far_gt_pts, near_gt_pts = self.get_ground_truth()
        if self.is_homography:
            return symmetricTransferError(self.H, far_gt_pts, near_gt_pts)
            # return self.near_to_far_error()
        elif self.is_fundamental:
            return fundMatReprojectionError(self.F, far_gt_pts, near_gt_pts)

    def near_to_far_error(self):
        if not self.is_homography:
            raise NotImplementedError()
        far_gt_pts, near_gt_pts = self.get_ground_truth()
        near_to_far_errs = transferErrors(np.linalg.inv(self.H), near_gt_pts,
                                          far_gt_pts)
        return np.mean(near_to_far_errs)

    def gt_scale_changes(self):
        far_gt_pts, near_gt_pts = self.get_ground_truth()
        return computeScaleChanges(far_gt_pts, near_gt_pts)

    def gt_inlier_scale_changes(self, threshold):
        if self.is_failure:
            return [0]
        gt_errors = self._separate_gt_errors()
        gt_pairs = list(zip(self.get_ground_truth()))
        inlier_matches = [p for p, e in zip(gt_pairs, gt_errors)
                          if e < threshold]
        if len(inlier_matches) < 2:
            return [0]
        far_gt_inliers, near_gt_inliers = list(zip(*inlier_matches))
        return computeScaleChanges(far_gt_inliers, near_gt_inliers)

    def inlier_scale_changes(self):
        return computeScaleChanges(self.far_inlier_pts, self.near_inlier_pts)

    def get_ground_truth(self):
        gtDict, _ = getGroundTruth(self.dataset, self.far_image_idx,
                                   self.near_image_idx, self.img_shape)
        return (np.array(gtDict[self.far_image_idx]),
                np.array(gtDict[self.near_image_idx]))



def analyzeMatches(matches):
    print((len(matches), 'matches found over', \
        len(np.unique(matches[:, 0])), 'far objects and', \
        len(np.unique(matches[:, 1])), 'near objects'))
    # of the objects that occur more than once, how many are inconsistent?
    leftFreqs = np.bincount(matches[:, 0])
    multiObjsLeft = np.where(leftFreqs > 1)[0]
    multiLeftObjMatchCounts = []
    for lObj in multiObjsLeft:
        lObjMatches = len(set([m[1] for m in matches if m[0] == lObj]))
        multiLeftObjMatchCounts.append(lObjMatches)
    print(('num. right objects matched to left objects:', multiLeftObjMatchCounts))

    rightFreqs = np.bincount(matches[:, 1])
    multiObjsRight = np.where(rightFreqs > 1)[0]
    multiRightObjMatchCounts = []
    for rObj in multiObjsRight:
        rObjMatches = len(set([m[0] for m in matches if m[1] == rObj]))
        multiRightObjMatchCounts.append(rObjMatches)
    print(('num. left objects matched to right objects:', \
           multiRightObjMatchCounts))


def voteOnMatches(matches, requireConsensus=False):
    winningMatches = []
    remainingMatches = []
    # do left, then right
    leftFreqs = np.bincount(matches[:, 0])
    rightFreqs = np.bincount(matches[:, 1])
    # add all matches between objects that participate in only that match
    for i, j in matches:
        if leftFreqs[i] == 1 and rightFreqs[j] == 1:
            winningMatches.append((i, j))
        else:
            remainingMatches.append((i, j))

    # add most common match of objects involved in more than one match
    leftBestMatches = []
    leftObjects = set([i for i, _ in remainingMatches])
    for leftObject in leftObjects:
        leftObjMatches = [j for i, j in remainingMatches if leftObject == i]
        # determine frequency of each match
        leftObjMatchFreqs = np.bincount(leftObjMatches)
        if requireConsensus and np.count_nonzero(leftObjMatchFreqs) > 1:
            continue
        # find best match
        bestMatch = np.argmax(leftObjMatchFreqs)
        bestFreq = leftObjMatchFreqs[bestMatch]
        if np.count_nonzero(leftObjMatchFreqs == bestFreq) == 1:
            leftBestMatches.append((leftObject, bestMatch))

    rightObjects = set([j for _, j in leftBestMatches])
    for rightObject in rightObjects:
        rightObjMatches = [i for i, j in leftBestMatches if rightObject == j]
        rightObjMatchFreqs = np.bincount(rightObjMatches)
        bestMatch = np.argmax(rightObjMatchFreqs)
        bestFreq = rightObjMatchFreqs[bestMatch]
        if np.count_nonzero(rightObjMatchFreqs == bestFreq) == 1:
            winningMatches.append((bestMatch, rightObject))
    # TODO alternately, only accept double matches...probably won't work

    return np.array(winningMatches)


def extractScenesFromDataset(objEngine, dataset, reuse=None):
    imgs = [cv2.imread(path) for path in dataset.images]
    imgBoxes = {}
    if reuse:
        boxPickleFilename = os.path.join(dataset.dir, reuse + '.pkl')
        if hasattr(objEngine, 'compute') and os.path.isfile(boxPickleFilename):
            # load pre-computed boxes
            with open(boxPickleFilename, 'rb') as f:
                imgBoxes = pickle.load(f)

    scenes = []
    for path in dataset.images:
        img = cv2.imread(path)
        imgName = os.path.basename(path)
        if isinstance(objEngine, CachedFeatureReader):
            scene = objEngine.read_cached_features_as_scene(path)
            scene.image = img
            imgBoxes[imgName] = list(zip(scene.boxes, scene.masks))
        else:
            if imgName in imgBoxes:
                # use pre-computed boxes to compute features
                boxes, masks = list(zip(*(imgBoxes[imgName])))
                scene = objEngine.getSceneFromBoxes(img, boxes, masks)
            else:
                # compute boxes from scratch and store them
                scene = objEngine.detectAndComputeScene(img)
                imgBoxes[imgName] = list(zip(scene.boxes, scene.masks))
        scenes.append(scene)

    # save computed boxes
    if reuse:
        with open(boxPickleFilename, 'wb') as f:
            pickle.dump(imgBoxes, f)
    return scenes


def convertToHomogeneous(pts):
    homogPts = np.ones((pts.shape[0], 3))
    homogPts[:, :2] = pts
    return homogPts


def symmetricTransferError(H, pts1, pts2):
    """
    Arguments:
    -- H: A 3x3 homography matrix from image 1 to 2, such that p2 = H * p1
    -- pts1: An Nx2 or Nx3 array of points in image 1.
    -- pts2: An Nx2 or Nx3 array of corresponding points in image 2.
    """
    errs1To2 = transferErrors(H, pts1, pts2)
    errs2To1 = transferErrors(np.linalg.inv(H), pts2, pts1)
    return sum(errs1To2) + sum(errs2To1)


def transferErrors(H, pts1, pts2):
    """
    Arguments:
    -- H: A 3x3 homography matrix from image 1 to 2, such that p2 = H * p1
    -- pts1: An Nx2 or Nx3 array of points in image 1.
    -- pts2: An Nx2 or Nx3 array of corresponding points in image 2.
    """
    pts1 = np.array(pts1)
    if pts1.shape[1] == 2:
        pts1 = convertToHomogeneous(pts1)
    pts2 = np.array(pts2)
    if pts2.shape[1] == 2:
        pts2 = convertToHomogeneous(pts2)

    estPts2 = np.dot(H, pts1.T).T
    estPts2_homogeneous = (estPts2.T / estPts2[:, 2]).T
    euclideanErrs = np.linalg.norm(estPts2[:, :2] - pts2[:, :2], axis=1)
    # print 'mean error:', np.mean(euclideanErrs)
    return euclideanErrs


def fundMatReprojectionError(F, pts1, pts2):
    """
    Based on the Fundamental-Matrix checking method from ORB-SLAM 2.

    Arguments:
    -- F: A 3x3 fundamental matrix from image 1 to 2, such that p2*F*p1 = 0
    -- pts1: An Nx2 or Nx3 array of points in image 1.
    -- pts2: An Nx2 or Nx3 array of corresponding points in image 2.
    """
    pts1 = np.array(pts1)
    if pts1.shape[1] == 2:
        # Nx3
        pts1 = convertToHomogeneous(pts1)
    pts2 = np.array(pts2)
    if pts2.shape[1] == 2:
        # Nx3
        pts2 = convertToHomogeneous(pts2)

    def computeReprojError(pts, reprojPts):
        numerators = [np.dot(u, v)**2 for u, v in zip(pts, reprojPts)]
        denominators = [u[0]**2 + u[1]**2 for u in reprojPts]
        return np.array(numerators) / np.array(denominators)

    # take the transpose to make it Nx3
    reprojPts1 = np.dot(F, pts1.T).T
    errors2 = computeReprojError(pts2, reprojPts1)
    reprojPts2 = np.dot(pts2, F)
    errors1 = computeReprojError(pts1, reprojPts2)

    totalError = np.sum(errors1) + np.sum(errors2)
    return totalError


def computeScaleChanges(matchPtsFar, matchPtsNear):
    ratios = []
    # for each pair of points:
    pairs = [(i, i+1) for i in range(len(matchPtsFar) - 1)]
    for i in range(len(matchPtsFar)):
        for j in range(i + 1, len(matchPtsFar)):
            farDist = np.linalg.norm(matchPtsFar[i] - matchPtsFar[j])
            nearDist = np.linalg.norm(matchPtsNear[i] - matchPtsNear[j])
            if farDist > 0 and nearDist > 0:
                # might happen if some points are matched twice
                ratios.append(nearDist / farDist)

    return ratios


def getGroundTruth(dataset, imgIndex1, imgIndex2, img_shape=None):
    if img_shape is None:
        sample_img = cv2.imread(dataset.images[0])
        img_shape = (sample_img.shape[1], sample_img.shape[0])
    img_shape = np.array(img_shape)
    gtYamlFilename = '{}_to_{}.yaml'.format(imgIndex1, imgIndex2)
    gtYamlPath = os.path.join(dataset.dir, gtYamlFilename)
    with open(gtYamlPath, 'r') as f:
        gtDict = yaml.load(f)
    gtIndexDict = {int(k.partition('.')[0]): gtDict[k] for k in gtDict}
    # convert to absolute image coordinates
    for k in gtIndexDict:
        abs_points = []
        for point in gtIndexDict[k]:
            abs_point = list(map(int, list(point * img_shape)))
            abs_points.append(abs_point)
        gtIndexDict[k] = abs_points
    return gtIndexDict, img_shape


def matchOverDataset(dataset, scenes, metric='cosine', ransacThreshold=None,
                     ratioThreshold=False, planar=True, showNow=True,
                     subFeatureMetric=None, name=''):
    imgLabels = [os.path.basename(path)[0] for path in dataset.images]
    pairIdxs = getUnorderedPairsFromList(dataset.images)
    results = []
    for i, j in pairIdxs:
        ptMatchTuple = loc.getPointMatchesFromObjects(scenes[i].objects,
                                                      scenes[j].objects,
                                                      metric, subFeatureMetric,
                                                      ratioThreshold=
                                                      ratioThreshold)
        pts1, pts2, objMatches = ptMatchTuple
        img_shape = (scenes[i].image.shape[1], scenes[i].image.shape[0])

        # compute homography/fundamental matrix from the matches
        fake_focal = 1
        fake_pp = (0.5, 0.5)
        try:
            if planar:
                loc.findTransformFrom2dMatches(pts1, pts2, fake_focal, fake_pp,
                                               ransacThreshold)
                # if the above didn't fail, get the matrix and used matches
                M, usedMatches = loc.findHomographyFromMatches(
                    pts1, pts2, ransacThreshold)
                result_type = TYPE_HOMOG
            else:
                loc.findTransformFrom3dMatches(pts1, pts2, fake_focal, fake_pp,
                                               ransacThreshold)
                # if the above didn't fail, get the matrix and used matches
                M, usedMatches = loc.findFundamentalMatFromMatches(
                    pts1, pts2, ransacThreshold)
                result_type = TYPE_FUND
            result = Result(dataset, i, j, img_shape, pts1, pts2, M,
                            usedMatches, result_type)
            results.append(result)
        except loc.LocalizationFailure as e:
            print(e)
            results.append(Result.FailureResult(dataset, i, j, img_shape))
            continue

    return results


def getUnorderedPairsFromList(theList, indices=True):
    pairs = []
    for i in range(len(theList)):
        for j in range(i + 1, len(theList)):
            if indices:
                pairs.append((i, j))
            else:
                pairs.append((theList[i], theList[j]))
    return pairs


def estimateFromGroundTruth(datasets, picklePath, planar=True,
                            ransacThreshold=6):
    """
    Run experiments using the ground truth as the point identifications.
    """
    baseName = 'Ground truth'
    results = []
    for dataset in datasets:
        datasetName = os.path.basename(dataset.dir)
        pairIdxs = getUnorderedPairsFromList(dataset.images)
        for i, j in pairIdxs:
            # read in the ground truth
            gtIndexDict, imgShape = getGroundTruth(dataset, i, j)
            iPts = np.array(gtIndexDict[i])
            jPts = np.array(gtIndexDict[j])
            if planar:
                # find homography using all points
                # use RANSAC, since all points aren't necessarily planar...
                H, mask = cv2.findHomography(iPts, jPts, cv2.RANSAC,
                                             ransacThreshold)
            else:

                F, mask = cv2.findFundamentalMat(iPts, jPts, cv2.FM_8POINT)

            matches = [(ptIdx, ptIdx) for ptIdx, _ in enumerate(iPts)]
            inlierMatches = np.array(matches)[np.nonzero(mask.flatten())]
            if planar:
                result = Result(dataset, i, j, imgShape, iPts, jPts, H,
                                inlierMatches)
            else:
                result = Result(dataset, i, j, imgShape, iPts, jPts, F,
                                inlierMatches, result_type=TYPE_FUND)
            results.append(result)

    # pickle the results
    if not picklePath:
        picklePath = 'gt_real_world.pkl'

    with open(picklePath, 'w') as f:
        dummyBaseConfig = {'localization': {'planar': planar},
                           'name': 'ground truth'}
        dummyActualConfig = dummyBaseConfig
        outputTuple = (dummyBaseConfig, [(dummyActualConfig, results)])
        pickle.dump(outputTuple, f)


def runRealWorldExperiments(baseConfig, datasets, showNow, reuse, picklePath,
                            device):
    """
    Run experiments.
    """
    baseName = baseConfig['name']
    locConfigs = expu.expandConfigToParamGrid(baseConfig['localization'])
    extConfigs = expu.expandConfigToParamGrid(baseConfig['extractor'])
    objEngine = None
    allResults = {}
    allConfigs = {}
    pbar = tqdm(total=len(extConfigs) * len(datasets) * len(locConfigs))
    for extConfig in extConfigs:
        if objEngine is not None:
            del objEngine
        objEngine = expu.getObjectExtractorFromConfig(extConfig, device)
        for dataset in datasets:
            scenes = extractScenesFromDataset(objEngine, dataset, reuse)
            for locConfig in locConfigs:
                configKey = baseConfig['name']
                if extConfig['name']:
                    configKey += '_' + extConfig['name']
                if locConfig['name']:
                    configKey += '_' + locConfig['name']

                if configKey not in allResults:
                    cfg = {'localization': locConfig,
                           'extractor': extConfig,
                           'name': configKey}
                    allResults[configKey] = (cfg, [])

                name = ''.join([baseName, extConfig['name'],
                                locConfig['name']])
                kwargs = locConfig.copy()
                kwargs['name'] = name
                results = matchOverDataset(dataset, scenes, showNow=showNow,
                                           **kwargs)
                allResults[configKey][1].extend(results)
                pbar.update(1)

    pbar.close()
    # convert results from a dict to a list
    allResults = [v for _, v in list(allResults.items())]
    # pickle the results
    if not picklePath:
        picklePath = baseConfig['name'] + '_real_world.pkl'
    with open(picklePath, 'wb') as f:
        pickle.dump((baseConfig, allResults), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true',
                        help='Display results immediately, instead of saving.')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--poster', action='store_true',
                     help='Only run on the poster test set.')
    grp.add_argument('--indoor', action='store_true',
                     help='Only run on the indoor sets.')
    parser.add_argument('--reuse', help='\
Reuse pre-computed bounding boxes, and save new bounding boxes for reuse.')
    parser.add_argument('--clearboxes', action='store_true', help='\
Delete existing pre-computed bounding boxes of type defined by --reuse arg.')
    parser.add_argument('-p', '--pickle', help='Name of output pickle file.')
    parser.add_argument('--cheat', action='store_true', help='\
Use the ground truth to compute the homographies.')
    parser = expu.addCommonArgs(parser)
    args = parser.parse_args()

    # select dataset
    if args.poster:
        datasets = [consts.posterTest]
    elif args.indoor:
        datasets = [consts.labDoor, consts.posterTest]
    else:
        datasets = consts.outdoorDatasets

    if args.cheat:
        estimateFromGroundTruth(datasets, args.pickle, args.planar)
        print('Done.')
        exit(0)

    if args.clearboxes:
        if not args.reuse:
            raise ValueError(
                'User requested boxes be cleared, but did not specify which \
                ones with --reuse!')
        else:
            for dataset in datasets:
                boxPickleFilename = os.path.join(dataset.dir,
                                                 args.reuse + '.pkl')
                if os.path.exists(boxPickleFilename):
                    os.remove(boxPickleFilename)

    baseConfigs = expu.getConfigsFromArgs(args)
    if len(baseConfigs) > 1:
        raise ValueError("Only one config file at a time, please!")
    print(('base configs:', baseConfigs))
    device = expu.getDeviceFromArgs(args)
    for baseConfig in baseConfigs:
        # run experiment for each config file the user supplies.
        runRealWorldExperiments(baseConfig, datasets, args.show, args.reuse,
                                args.pickle, device)
