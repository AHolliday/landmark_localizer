# external imports
import cv2
import numpy as np
from collections import defaultdict
import scipy.spatial.distance as distance

# internal imports
from . import myCvTools
from . import constants as consts
from . import geometryUtils as geo

np.seterr(all='raise')

# same value used in opencv's cheirality checking
MAX_STABLE_DIST = 50


class LocalizationFailure(Exception):
    """Exception for failures in the localization pipeline."""
    pass


# Localization functions.


def findCameraTransformFromObjects(objs1, objs2, focal, pp, metric='cosine',
                                   ransacThreshold=None, planar=False,
                                   ratioThreshold=None, subFeatureMetric=None,
                                   cheat=False):
    # raise NotImplementedError("This method was abandoned with its code in a \
    #                           broken state.  It does not handle the 'planar'\
    #                           argument properly.")
    """
    Finds possible transforms between cameras based on pairs of point features.

    If the user specifies that the points are planar, do this by finding a \
    homography and decomposing it to possible transforms.  Otherwise, do this \
    by finding the essential matrix and decomposing that.  Either way, \
    possible transforms are filtered via cheirality checking.

    Arguments
    objs1 -- an ordered collection of objects first scene
    objs2 -- an ordered collection of objects in the second scene
    focal -- the focal length, in pixels, of the camera
    pp -- the image center (cx, cy), in pixels, of the camera

    Keyword arguments
    metric -- a value for the arg 'metric' in distance.cdist.
    ransacThreshold -- threshold to use in the robust transform-finding \
    process.  If not provided, LMEDS is used instead of RANSAC.
    planar -- Set to True if the points come from a planar environment.
    ratioThreshold -- if provided, boxes whose side ratios differ by more \
    than this amount will not be considered for matching.

    Returns
    M --  If planar = True, the computed homography matrix, otherwise \
    the computed essential matrix.
    Rp21s -- A list of possible rotation matrices from the orientation of \
    cam 2 to that of cam 1.
    Tp21s -- A list of possible positions of cam 1's center in cam 2's \
    frame, corresponding to the rotations in Rp21s.
    N1s -- If planar = True, a list of possible normals of the plane in \
    cam 1's frame of reference, corresponding to the rotations in Rp21s. \
    If planar = False, a list of Nones.
    pts3dSets -- A list of collections of valid 3d world points in cam 1's \
    frame, where the points in pts3dSets[i] are triangulated from Rp21s[i] \
    and Tp21s[i].  Each collection is a numpy array with each point as a row.
    usedMatchesSets -- A list of collections of pairs of indices in points1 \
    and points2.  usedMatchesSets[i][j] is the pair of indices of the 2d \
    points that triangulate to the 3d point pts3dSets[i][j].
    pts1 -- the points in the first camera
    pts2 -- the points in the second camera
    """
    # set up variables.
    K = geo.buildIntrinsicMatrix(focal, focal, *pp)

    # perform matching of boxes by features and ratios
    pts1, pts2, objMs = getPointMatchesFromObjects(objs1, objs2, metric,
                                                    subFeatureMetric,
                                                    ratioThreshold)

    # Estimate the transform and triangulate the points based on it.
    # try:
    #     results2d = findTransformFrom2dMatches(pts1, pts2, matches, focal, pp,
    #                                             ransacThreshold)
    #     H, R_21s, T_21s, N1s, pts3dSets, usedMatchesSets, scores2d = results2d
    #     bestScore2d = max(scores2d)
    # except LocalizationFailure as lf:
    #     print lf
    #     results2d = None
    #     bestScore2d = -np.inf

    try:
        results3d = findTransformFrom3dMatches(pts1, pts2, focal, pp,
                                                ransacThreshold)
        # if results3d is None:
        #     return (None,) * 6
        E, Rp21, Tp21, usedMatches = results3d
        Mp21 = geo.buildTfMatrix(Rp21, Tp21)
        usedPts1 = pts1[usedMatches[:, 0]]
        usedPts2 = pts2[usedMatches[:, 1]]
        pts3d = geo.triangulatePoints(usedPts1, usedPts2, K, Mp21=Mp21)
        # discard projective coordinate and arrange points as rows
        pts3d = pts3d[:3, :].T

        score3d = getModelScore(usedPts1, usedPts2, pts3d, Rp21, Tp21, K,
                                ransacThreshold)
        if not score3d > 0:
            raise LocalizationFailure("Transform from E has score 0!")
    except LocalizationFailure as lf:
        # print(lf)
        results3d = None
        score3d = -np.inf
        if not planar:
            raise lf

    # if cheat and None not in (results2d, results3d):
    #     M = [H, E]
    #     R_21s.append(R_21)
    #     T_21s.append(T_21)
    #     N1s.append(None)
    #     pts3dSets.append(pts3d)
    #     usedMatchesSets.append(usedMatches)
    #
    # elif planar:
    #     M = H
    #     # return the homography
    if not planar:
        # Package values for returning.  No normal vector exists for a
        # non-planar environment.
        # T_21 = - np.dot(R_21, T_12)
        M = E
        pts3dSets = [pts3d]
        N1s = [None]
        Rp21s = [Rp21]
        Tp21s = [Tp21]
        usedMatchesSets = [usedMatches]

    if subFeatureMetric:
        usedMatchesSets = [objMs] * len(Tp21s)

    return M, Rp21s, Tp21s, N1s, pts3dSets, usedMatchesSets, usedPts1, usedPts2


def getPointMatchesFromObjects(objs1, objs2, metric='cosine',
                               subFeatureMetric='euclidean',
                               ratioThreshold=False, returnDists=False,
                               **kwargs):
    boxMask = np.ones((len(objs1), len(objs2))).astype(bool)
    if ratioThreshold:
        expThresh = np.exp(ratioThreshold)
        for i1, obj1 in enumerate(objs1):
            for i2, obj2 in enumerate(objs2):
                if computeAspectRatioScore(obj1.box, obj2.box) > expThresh:
                    boxMask[i1, i2] = False

    # import pdb; pdb.set_trace()
    objMatches = np.zeros((0, 2), dtype=int)
    matchDists = []

    # match each feature layer separately, and concatenate the matches.
    descs1 = [ob.descriptor.flatten() for ob in objs1]
    descs2 = [ob.descriptor.flatten() for ob in objs2]
    objMatches, matchDists = matchByBruteForce(descs1, descs2, metric,
                                               crossCheck=True, mask=boxMask)
                                                 # augMatrix=shapeScoreMatrix)
    pts1, pts2 = [], []

    haveSubFeatures = any([len(o.getSubFeatures()) > 0 for o in objs1 + objs2])
    if haveSubFeatures and subFeatureMetric:
        # compute matching on the subfeatures, rather than the features.
        mObjs1, mObjs2 = list(zip(*[(objs1[i], objs2[j])
                                    for i, j in objMatches]))
        matchSubFeaturePts = matchSubFeatures(mObjs1, mObjs2, subFeatureMetric,
                                              crossCheck=True)

        if len(matchSubFeaturePts) > 0:
            subFeaturePts1, subFeaturePts2 = list(zip(*matchSubFeaturePts))
            # include sub-feature points
            pts1 += subFeaturePts1
            pts2 += subFeaturePts2
        # matches = np.array([(i, i) for i in range(len(subFeaturePts1))])
    # else:
    if len(pts1) < 5:
        # these are generally imprecise, so avoid them.
        for i, j in objMatches:
            # add the object match center points
            pts1.append(objs1[i].center)
            pts2.append(objs2[j].center)

    # all the points provided are matches.
    ret_tpl = np.array(pts1), np.array(pts2), objMatches
    if returnDists:
        return ret_tpl + (matchDists,)
    else:
        return ret_tpl


def findHomographyFromMatches(points1, points2, ransacThreshold, matches=None):
    if (matches and len(matches) < 4) or len(points1) < 4:
        raise LocalizationFailure("Not enough matches for homography.")
    H, usedMatches = _findMatFromMatches_helper(points1, points2,
                                                ransacThreshold,
                                                cv2.findHomography, matches)
    try:
        np.linalg.inv(H)
    except np.linalg.linalg.LinAlgError:
        raise LocalizationFailure('Homography is singluar!')
    return H, usedMatches


def findFundamentalMatFromMatches(points1, points2, ransacThreshold,
                                  matches=None):
    if matches is not None and len(matches) < 8:
        raise LocalizationFailure("Not enough matches for fundamental matrix.")
    F, usedMatches = _findMatFromMatches_helper(points1, points2,
                                                ransacThreshold,
                                                cv2.findFundamentalMat,
                                                matches)
    return F, usedMatches


def _findMatFromMatches_helper(points1, points2, ransacThreshold,
                               findMatFunction, matches=None):
    mp1 = np.array(points1)
    mp2 = np.array(points2)
    if matches:
        matchInds1, matchInds2 = list(zip(*matches))
        mp1 = mp1[matchInds1]
        mp2 = mp2[matchInds2]
    else:
        # matches were not provided, so assume the points are in matched order
        matches = np.array([(ii, ii) for ii in range(len(points1))])

    # find the homography given the matching points.
    # not sure why we do this reshape, but it's what they do in the example...
    mp1_reshaped = np.float32(mp1).reshape(-1, 1, 2)
    mp2_reshaped = np.float32(mp2).reshape(-1, 1, 2)
    if ransacThreshold is None or ransacThreshold == 0:
        M, mask = findMatFunction(mp1_reshaped, mp2_reshaped, cv2.LMEDS)
    else:
        M, mask = findMatFunction(mp1_reshaped, mp2_reshaped, cv2.RANSAC,
                                  ransacThreshold)

    if M is None or np.inf in M or -np.inf in M:
        raise LocalizationFailure(findMatFunction.__name__ + " failed to \
compute a matrix!  The points must have been bad.")

    usedMatches = matches[np.nonzero(mask.flatten())]
    return M, usedMatches


def findTransformFrom2dMatches(points1, points2, focal, pp,
                                ransacThreshold=None, matches=None):
    """
    Computes homography and estimates the best transform from the matches.

    Opencv follows the homography-decomposition method described in "Deeper
    understanding of the homography decomposition for vision-based control"
    by Malis and Vargas.  In this document, they assume the plane normal
    points *away* from the camera; so it should have positive z component.
    They define H as:

    H21 = Rp12 + dot_product((t112 / d2), n_2)

    Where:
    Rp12 is the rotation from cam 1's pose to cam 2's pose,
    t112 is the translation from cam 1 to cam 2 in cam 1's frame,
    d2 is the distance of cam 2 from the plane,
    n_2 is the normal vector of the plane in cam 2's frame.

    In fact though, opencv gives back H12, so the values we get are:

    H12 = Rp21 + dot_product((t221 / d1), n1)
    """
    # Compute a homography from the matches and decompose it to possible
    # transforms.

    if type(points1) is not np.ndarray:
        points1 = np.array(points1)
    if type(points2) is not np.ndarray:
        points2 = np.array(points2)
    H12, usedMatches = findHomographyFromMatches(points1, points2,
                                                 ransacThreshold, matches)
    usedPts1 = points1[usedMatches[:, 0]]
    usedPts2 = points2[usedMatches[:, 1]]
    K = geo.buildIntrinsicMatrix(focal, focal, *pp)
    count, Rp21s, t221s, N1s = cv2.decomposeHomographyMat(H12, K)
    if np.any(np.isnan(Rp21s)) or np.any(np.isnan(t221s)):
        raise LocalizationFailure('NaNs returned when decomposing homography!')

    # perform cheirality check in the same way as opencv's recoverPose.
    # For each transform, triangulate the world points in both frames.
    # Reject points at negative depth and at unstable normalized depth from
    # either point of view, and record the surviving points in cam 1's frame
    # (sans homogeneous coordinate) to be returned later.
    validMasks = np.ones((len(N1s), len(usedMatches)), dtype=bool)
    usedMatchesSets = []
    pts3dSets = []
    scores = []
    for i, (Rp21, t221, mask) in enumerate(zip(Rp21s, t221s, validMasks)):
        Rp12 = Rp21.T
        t112 = -np.dot(Rp12, t221)
        Mp12 = geo.buildTfMatrix(R=Rp12, T=t112)
        worldPoints_cam1Frame = geo.triangulatePoints(usedPts1, usedPts2,
                                                      K, Mp12=Mp12)
        Mc_1to2 = geo.buildTfMatrix(R=Rp21, T=t221)
        worldPoints_cam2Frame = np.dot(Mc_1to2, worldPoints_cam1Frame)
        mask &= ((worldPoints_cam1Frame[2, :] > 0) &
                 (worldPoints_cam1Frame[2, :] < MAX_STABLE_DIST) &
                 (worldPoints_cam2Frame[2, :] > 0) &
                 (worldPoints_cam2Frame[2, :] < MAX_STABLE_DIST))
        maskAsIdxList = np.where(mask)[0]
        # return points arranged as rows, easier to work with in python
        inlierWorldPts = worldPoints_cam1Frame[:3, maskAsIdxList].T
        inlierMatches = usedMatches[maskAsIdxList]
        score = getModelScore(points1[inlierMatches[:, 0]],
                              points2[inlierMatches[:, 1]],
                              inlierWorldPts, Rp21, t221, K, ransacThreshold)
        scores.append(score)
        pts3dSets.append(inlierWorldPts)
        usedMatchesSets.append(inlierMatches)

    # Return all transforms that are tied for most valid.
    validCounts = np.sum(validMasks, axis=1)
    # print 'validity counts:', validCounts, '(of', validMasks.shape[1], ')'
    # bestCount = np.max(validCounts)
    bestScore = np.max(scores)
    if not bestScore > 0:
        raise LocalizationFailure("All homography transforms have score 0.")
    bestCounts = validCounts[np.argmax(scores)]
    if bestCounts < 4:
        raise LocalizationFailure("Homography has < 4 inliers.")
    # if np.argmax(validCounts) != np.argmax(scores):
    #     print 'WARNING: valid counts and scores differ.', validCounts, scores
    # bestIdxs = np.where(validCounts == bestCount)[0]
    bestIdxs = np.where(np.array(scores) == bestScore)[0]

    zippedLists = list(zip(Rp21s, t221s, N1s, pts3dSets, usedMatchesSets, scores))
    bestTuples = list(zip(*[zippedLists[i] for i in bestIdxs]))
    Rp21s, t221s, N1s, pts3dSets, usedMatchesSets, scores = list(map(list,
                                                                bestTuples))
    # print 'of', len(scores), 'returned scoreforms', np.argmax(scores), 'is best'
    return H12, Rp21s, t221s, N1s, pts3dSets, usedMatchesSets, scores


def findTransformFrom3dMatches(points1, points2, focal, pp,
                                ransacThreshold=None, matches=None):
    """
    Finds the transform that takes camera 1 to camera 2 for non-planar points.

    Returns:
    -- E: the essential matrix relating camera 1 and camera 2.
    -- Rp21, the rotation from camera 2's frame to camera 1's frame.
    -- Tp21, the position of camera 1's center in camera 2's frame.
    Together Rp21 and Tp21 form Mp21.
    """
    # find the matching points between images.
    if (matches and len(matches) < 5) or len(points1) < 5 or len(points2) < 5:
        raise LocalizationFailure("Not enough matches to find essential matrix.")
        # return None

    mp1 = np.array(points1)
    mp2 = np.array(points2)
    if matches:
        matchInds1, matchInds2 = list(zip(*matches))
        mp1 = mp1[matchInds1]
        mp2 = mp2[matchInds2]
    else:
        # matches were not provided, so assume the points are in matched order
        matches = np.array([(ii, ii) for ii in range(len(points1))])
    #
    #
    # matchInds1, matchInds2 = list(map(list, list(zip(*matches))))
    # mp1 = np.array(points1)[matchInds1]
    # mp2 = np.array(points2)[matchInds2]

    if ransacThreshold is None or ransacThreshold == 0:
        E, mask = cv2.findEssentialMat(mp1, mp2, method=cv2.LMEDS,
                                       focal=focal, pp=pp)
    else:
        E, mask = cv2.findEssentialMat(mp1, mp2, method=cv2.RANSAC,
                                       focal=focal, pp=pp,
                                       threshold=ransacThreshold)
    if E is None or E.shape != (3, 3):
        raise LocalizationFailure("findEssentialMat failed to compute an essential matrix!")
        # return None

    # Recover the pose using only use essential matrix inliers.
    usedMatchIdxs = np.nonzero(mask.flatten())
    usedOrigMatches = matches[usedMatchIdxs]
    # usedMatchInds1, usedMatchInds2 = map(list, zip(*usedMatches))
    mp1 = mp1[usedMatchIdxs]
    mp2 = mp2[usedMatchIdxs]
    _, Rp21, Tp21, rpMask = cv2.recoverPose(E, mp1, mp2, focal=focal, pp=pp)
    usedOrigMatches = usedOrigMatches[np.where(rpMask.flatten())]

    return E, Rp21, Tp21, usedOrigMatches


def filterByRatioSimilarity(boxes1, boxes2, threshold=0.2):
    def getRatios(bs):
        w = np.array([b[2] - b[0] for b in bs]).astype(float)
        h = np.array([b[3] - b[1] for b in bs]).astype(float)
        ratios = np.maximum(w / h, h / w)
        if len(ratios.shape) == 1:
            ratios = ratios[:, np.newaxis]
        return ratios

    ratios1, ratios2 = list(map(getRatios, [boxes1, boxes2]))
    ratioDiff = distance.cdist(ratios1, ratios2)
    mask = ratioDiff < threshold
    return mask


def matchSubFeatures(objs1, objs2, metric='euclidean', crossCheck=True):
    # to work around how everything takes a list of matches:
    matches = []
    bfMatcher = cv2.BFMatcher(crossCheck=crossCheck)
    for obj1, obj2 in zip(objs1, objs2):
        if len(obj1.getSubFeatures()) > 0 and len(obj2.getSubFeatures()) > 0:
            _, descs1 = list(zip(*obj1.getSubFeatures()))
            _, descs2 = list(zip(*obj2.getSubFeatures()))

            obj_pair_matches = [(obj1.subFeatIdxs[mm.queryIdx],
                                 obj2.subFeatIdxs[mm.trainIdx])
                                for mm in bfMatcher.match(np.array(descs1),
                                                          np.array(descs2))]
            matches += obj_pair_matches

    allSubFeatPts1 = [pt.pt for pt, _ in objs1[0].allGlobalFeatures]
    allSubFeatPts2 = [pt.pt for pt, _ in objs2[0].allGlobalFeatures]
    ptMatches = [(allSubFeatPts1[i], allSubFeatPts2[j])
                 for i, j in matches]
    return ptMatches


def matchSubFeatures1To1(objs1, objs2, metric='euclidean', obj_dists=None,
                         use_sub=True, crossCheck=False,
                         filterZeroVectors=True):
    """This gives very bad results."""
    all_subs_1 = objs1[0].allGlobalFeatures
    all_subs_2 = objs2[0].allGlobalFeatures
    matches_1to2 = defaultdict(list)
    matches_2to1 = defaultdict(list)
    if not obj_dists:
        obj_dists = [1] * len(objs1)
        if not use_sub:
            raise ValueError('Must either provide obj_dists or use_sub=True!')

    # find all candidate matches for each SIFT feature
    for obj1, obj2, obj_dist in zip(objs1, objs2, obj_dists):
        if len(obj1.subFeatIdxs) > 0 and len(obj2.subFeatIdxs) > 0:
            _, descs1 = list(zip(*obj1.getSubFeatures()))
            _, descs2 = list(zip(*obj2.getSubFeatures()))
            matches, sub_dists = matchByBruteForce(descs1, descs2, metric,
                                                   crossCheck,
                                                   filterZeroVectors)
            for (ii, jj), sub_dist in zip(matches, sub_dists):
                dist = obj_dist
                if use_sub:
                    dist *= sub_dist
                matches_1to2[ii].append((jj, dist))
                matches_2to1[jj].append((ii, dist))

    # all possible matches have been found; now remove duplicates
    final_matches = []
    # we do cross-checking here.
    for ii, candidates in matches_1to2.items():
        candidates = sorted(candidates, key=lambda tpl: tpl[1])
        best_jj, best_ij_dist = candidates[0]
        # neither cross-check nor Lowe's; just take the best one
        final_matches.append(((ii, jj), best_ij_dist))

        # if crossCheck:
        #     cross_candidates = matches_2to1[best_jj]
        #     best_ii, _ = min(cross_candidates, key=lambda tpl: tpl[1])
        #     if best_ii == ii:
        #         final_matches.append(((ii, jj), best_ij_dist))
        # # if not cross-checking, use Lowe's ratio
        # elif best_ij_dist < candidates[1][1] * 0.7:
        #     final_matches.append(((ii, best_jj), best_ij_dist))

    matches, distances = list(zip(*final_matches))
    match_pts = [(all_subs_1[ii][0].pt, all_subs_2[jj][0].pt)
                 for ii, jj in matches]
    return match_pts, distances


def filterForBestMatches(matches, distances):
    """
    Arguments:
    matches -- a collection of match pairs in which each element may \
    participate in multiple pairs (should be indices or some other \
    type amenable to comparison).
    distances -- a quality measure of each match, where smaller = better.

    Returns:
    bestMatches -- a collection of match pairs in which each element \
    participates in at most one pair, the best of the pairs it appears in in \
    the original list.
    distances -- the distances of each match in bestMatches.
    """
    matchToDistDict = {}
    # matchesWithDists = list(zip(matches, distances))
    for match, dist in zip(matches, distances):
        match = tuple(match)
        if match in matchToDistDict:
            matchToDistDict[match] += dist
        else:
            matchToDistDict[match] = dist
    matchesWithDists = [(k, v) for k, v in list(matchToDistDict.items())]

    # sort in ascending order
    sortedMatchesWithDists = sorted(matchesWithDists, key=lambda x: x[1])
    bestMatchesLeft = []
    bestMatchesRight = []
    bestMatchDists = []
    for (left, right), dist in sortedMatchesWithDists:
        if left not in bestMatchesLeft and right not in bestMatchesRight:
            # one side participates in a better match, so don't add this.
            bestMatchesLeft.append(left)
            bestMatchesRight.append(right)
            bestMatchDists.append(dist)
    bestMatches = np.array(list(zip(bestMatchesLeft, bestMatchesRight)))
    return bestMatches, bestMatchDists


def matchByBruteForce(feats1, feats2, metric, crossCheck=False,
                      filterZeroVectors=True, mask=None, augMatrix=None,
                      silent=True):
    """
    Finds the set of matches between two vector sets by brute-force matching.

    Arguments:
    feats1 -- a ordered collection of flat feature vectors.
    feats2 -- a second ordered collection of flat feature vectors.

    Keyword args:
    metric -- a function that computes distance between vectors (cosine \
    distance by default.)
    crossCheck -- if true, perform cross-checking of results.
    filterZeroVectors -- remove features that are 0 everywhere
    mask -- a mask indicating which features to ignore
    augMatrix -- a len(feats1) x len(feats2) matrix to be multiplied by the
        distances

    Returns:
    - A list of pairs of indices in the two sets of matching vectors
    - A list of the distances between each pair
    """

    if len(feats1) == 0 or len(feats2) == 0:
        return np.zeros((0, 2)), np.array([])
    # Matches will be a dict during the function, for efficient lookup
    matches = {}
    if filterZeroVectors:
        if mask is None:
            mask = np.ones((len(feats1), len(feats2)), dtype=np.bool)
        feats1_are_zero = np.where([np.count_nonzero(_) == 0 for _ in feats1])
        mask[feats1_are_zero, :] = False
        feats2_are_zero = np.where([np.count_nonzero(_) == 0 for _ in feats2])
        mask[:, feats2_are_zero] = False

    # find closest point in set 2 to each point in set 1
    distances = distance.cdist(feats1, feats2, metric=metric)
    if augMatrix is not None:
        distances *= augMatrix
    if mask is not None:
        # ignore all pairs that are False in the mask
        distances[np.logical_not(mask)] = np.inf

    matches1to2 = np.argmin(distances, axis=1)

    if crossCheck:
        # find matches that are the same in both directions
        matches2to1 = np.argmin(distances, axis=0)
        matches = {i1: i2 for i1, i2 in enumerate(matches1to2)
                   if matches2to1[i2] == i1}
    else:
        matches = {i1: i2 for i1, i2 in enumerate(matches1to2)}

    # convert it to a list of tuples
    matches = [(i1, i2) for i1, i2 in list(matches.items())]
    matchDists = distances[tuple(zip(*matches))]

    # print some stats about it
    if not silent:
        print(('match dist mean:', np.mean(matchDists), \
            'median:', np.median(matchDists), \
            'std dev:', np.std(matchDists), \
            'max:', np.max(matchDists), \
            'min:', np.min(matchDists)))

    matches, matchDists = list(zip(*(sorted(zip(matches, matchDists.flatten()),
                                       key=lambda x: x[1]))))
    return np.array(matches), matchDists


def getModelScore(imgPts1, imgPts2, estWorldPts1, Rp21, t221, K, threshold):
    if estWorldPts1.size == 0:
        return 0.

    # project the world points to image 1
    rvec1, _ = cv2.Rodrigues(np.eye(3))
    tvec1 = np.zeros(3)
    projPts1, _ = cv2.projectPoints(estWorldPts1, rvec1, tvec1, K, None)
    reprojErrors1 = np.linalg.norm(imgPts1 - np.squeeze(projPts1), axis=1)
    scores1 = threshold - reprojErrors1
    scores1[scores1 < 0] = 0
    score1 = np.sum(scores1)

    # project the world points to image 2
    # rvec2, _ = cv2.Rodrigues(Rp21.T)
    rvec2, _ = cv2.Rodrigues(Rp21)
    # tvec2 = np.dot(Rp21.T, -t221)
    tvec2 = t221
    projPts2, _ = cv2.projectPoints(estWorldPts1, rvec2, tvec2, K, None)
    reprojErrors2 = np.linalg.norm(imgPts2 - np.squeeze(projPts2), axis=1)
    scores2 = threshold - reprojErrors2
    scores2[scores2 < 0] = 0
    score2 = np.sum(scores2)

    # total score is the sum of scores in both directions
    return score1 + score2


def computeShapeScore(obj1, obj2):
    """This is the shape similarity score used by Sunderhauf et al. in
    'Place Recognition with ConvNet Landmarks: Viewpoint-Robust,
    Condition-Robust, Training-Free'.  See equation 2 of that paper.

    ...but I doubt we'll ever use this, since it penalizes *absolute* shape,
    not aspect ratio.  So correct matches at large scale changes will be
    penalized by this!"""
    w_term = abs(obj1.width - obj2.width) / max(obj1.width, obj2.width)
    h_term = abs(obj1.height - obj2.height) / max(obj1.height, obj2.height)
    shape_score = np.exp(0.5 * (w_term + h_term))
    return shape_score


def computeAspectRatioScore(box1, box2):
    """Turns out this makes things worse too."""
    ratio1 = (box1[2] - box1[0]) / (box1[3] - box1[1])
    ratio2 = (box2[2] - box2[0]) / (box2[3] - box2[1])
    # apply exp to decrease penalty for small differences, relative to large ones
    shape_score = np.exp(abs(ratio1 - ratio2) / max(ratio1, ratio2))
    return shape_score
