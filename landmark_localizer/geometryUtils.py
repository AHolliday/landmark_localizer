import cv2
import math
import numpy as np
import scipy.spatial.distance as distance
from . import transformations

"""
A library module containing functions to peform various geometric operations.
In many cases, provides convenient wrappers for functions from other libraries.
"""

def cosineDistance(vec1, vec2):
    if np.all(vec1 == 0) and np.any(vec2 != 0):
        return 0.5
    if vec1 is None or vec2 is None:
        return 1
    else:
        return distance.cosine(vec1.flatten(), vec2.flatten())


def relativeDistance(pos1, pos2):
    """A measure of the distance between two vectors that ranges from 0 to 1.

    Unlike cosine distance, this captures differences in length as well as
    direction.
    """
    abs_err = np.linalg.norm(pos1 - pos2)
    pos1_norm = np.linalg.norm(pos1)
    if pos1_norm == 0:
        # not really sure what the correct behaviour is in this case...but
        # the below is a decent approximation
        return abs_err
    return abs_err / pos1_norm
    # return abs_err / (np.linalg.norm(pos1) + np.linalg.norm(pos2))


def rotationMatrixDistance(rotmat1, rotmat2):
    """A measure of the distance between two R matrices, ranging from 0 to 2.

    This computes the cosine distance between the quaternion representations of
    the two matrices.
    """
    q1 = rotMatrixToQuaternion(rotmat1)
    q2 = rotMatrixToQuaternion(rotmat2)
    dot_product = np.dot(q1, q2)
    return 1 - abs(dot_product)


def relativePoseDistance(pos1, rotmat1, pos2, rotmat2):
    """Given two rotations and translations, computes their distance as the
    sum of their relative positional distance and rotational distance, after
    normalizing both measures to have the same range."""
    pos_dist = relativeDistance(pos1, pos2)
    # ranges from 0 to 2, so divide by 2 to give it the same range as rot_dist
    rot_dist = rotationMatrixDistance(rotmat1, rotmat2) / 2
    return pos_dist + rot_dist


def get2dRotmatFromYaw(yaw):
    """Uses the math cos and sin functions for improved efficiency(?)"""
    return np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])


def getCenterFromBox(boxVector):
    """Returns the point at the center of the 2D box defined by 2 points.

    Arguments:
    -- boxVector: a vector of the form [xmin ymin xmax ymax ...]
    Returns:
    -- a numpy array of the form [xCenter, yCenter]
    """
    xmin, ymin, xmax, ymax = boxVector[:4]
    return np.array([(xmin + xmax) / 2.0, (ymin+ymax) / 2.0])


def getCornersFromBox(boxVector):
    """Given a bounding-box vector, returns the corner points.

    Arguments:
    -- boxVector: a vector of the form [xmin ymin xmax ymax ...]

    Returns:
    -- a 2D numpy array of corner points:
       [[xTopLeft,     yTopLeft],
        [xTopRight,    yTopRight],
        [xBottomLeft,  yBottomLeft],
        [xBottomRight, yBottomRight]]
    """
    xmin, ymin, xmax, ymax = boxVector[:4]
    corners = np.zeros((4, 2))
    corners[0] = [xmin, ymin]
    corners[1] = [xmax, ymin]
    corners[2] = [xmin, ymax]
    corners[3] = [xmax, ymax]
    return corners


def cvtToCvAxes(posVector):
    if posVector.size != 3:
        raise ValueError("Position must be a 3-vector!")
    return posVector[[1, 2, 0]] * (-1, -1, 1)


def rotMatrixToQuaternion(R):
    return transformations.quaternion_from_matrix(buildTfMatrix(R))


def buildTfMatrix(R=np.eye(3), T=np.zeros((3, 1))):
    outMat = np.eye(4)
    outMat[:3, :3] = R
    outMat[:3, 3] = T.flatten()
    return outMat


def buildIntrinsicMatrix(fx, fy, cx, cy):
    K = np.eye(3)
    # set focal length on the diagonal
    K[0, 0] = fx
    K[1, 1] = fy
    # set the principle point in the final column
    K[0, 2] = cx
    K[1, 2] = cy
    return K.astype(np.float32)


def buildProjectionMatrix(K, R=np.eye(3), T=np.zeros(3)):
    """
    Builds projection matrix from K, R, and T.

    If R and/or T are not provided, they are assumed to have values \
    corresponding to no rotation/translation.
    """
    P = np.zeros((3, 4))
    P[:, :3] = R
    if len(T.shape) > 1:
        T = T.flatten()
    P[:, 3] = T.flatten()
    P = np.dot(K, P)
    return P


def buildHomographyMatrix(R_21, T_21, N_1=np.array([0, 0, 1]), d=1,
                          K=np.eye(3)):
    """
    Get the perspective matrix resulting from a transform.

    Assumes that the transform is relative to an initial pose looking \
    directly at the plane along its normal vector at distance d=1 from the \
    plane.  Then:

    H_12 = (R_21 + (T_21*N_1^T) / d) = R - []
    [[T_xN_x, T_xN_y, T_xN_z]
     [T_yN_x, T_yN_y, T_yN_z]
     [T_zN_x, T_zN_y, T_zN_z]]

    Arguments
    R_21 -- the rotation matrix applied to camera 2 to change its \
    orientation to that of camera 1.
    T_21 -- the position of camera 1 in camera 2's coordinate frame.

    Keyword arguments
    N_1 -- the normal of the plane with respect to the first camera frame.
       If not provided, assumed to point directly away from camera1's center.
    d -- the distance of camera 1 from the plane.  Set to 1 if not provided.

    Returns
    H_12 -- a 3x3 matrix s.t. X2 = H_12 * X1, where X1 and X2 are the \
    positions of world point X in the frames of camera 1 and 2, respectively.
    """

    N_row = np.expand_dims(N_1.flatten(), axis=0)
    T_col = np.expand_dims(T_21.flatten(), axis=1)
    H_12 = R_21 + (np.dot(T_col, N_row) / d)
    # H_12 is such that X_2 = K * H_12 * inv(K) * X_1
    HPxSpace_12 = np.dot(np.dot(K, H_12), np.linalg.inv(K))
    return HPxSpace_12


def triangulatePoints(points1, points2, K, Mp21=None, Mp12=None):
    """
    """
    if Mp21 is None:
        if Mp12 is None:
            raise ValueError("Must provide one of Mp21 or M12!")
        Mc12 = np.linalg.inv(Mp12)
    else:
        Mc12 = Mp21
    if points1.size == 0 or points2.size == 0:
        return np.zeros((0, 4))

    P1 = np.dot(K, buildTfMatrix()[:3])
    P2 = np.dot(K, Mc12[:3])
    pts3d_projective = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    pts3d = pts3d_projective / pts3d_projective[3]
    return pts3d


def getTransformFromPoses(gtT01, gtRp01, gtT02, gtRp02):
    RcIC = np.array([[ 0,-1, 0],
                     [ 0, 0,-1],
                     [ 1, 0, 0]])
    gtRp01 = np.dot(RcIC, gtRp01)
    gtRp02 = np.dot(RcIC, gtRp02)

    gtRp12 = np.dot(gtRp01.T, gtRp02)
    gtRp21 = gtRp12.T
    gtT012 = gtT02 - gtT01
    gtT021 = -gtT012
    gtRc02 = gtRp02.T
    gtT21 = np.dot(gtRc02, gtT021)
    return gtRp21, gtT21


def evaluateEstimatesRePoses(gtT01, gtRp01, gtT02, gtRp02, R21s, T21s):
    # gtRc12 = np.dot(gtRp02, gtRp01.T)
    # this matrix rotates points from x-forward y-left z-up coordinates to
    # x-right y-down z-forward coordinates, which is what our Rs and Ts use.
    # We rotate the whole world's axes this way first.
    # RcIC = np.array([[ 0,-1, 0],
    #                  [ 0, 0,-1],
    #                  [ 1, 0, 0]])
    # gtRp01 = np.dot(RcIC, gtRp01)
    # gtRp02 = np.dot(RcIC, gtRp02)

    # gtRp12 = np.dot(gtRp01.T, gtRp02)
    # gtRp21 = gtRp12.T
    # gtT012 = gtT02 - gtT01
    # gtT021 = -gtT012
    # gtRc02 = gtRp02.T
    # gtT21 = np.dot(gtRc02, gtT021)
    # compute error of the result
    gtRp21, gtT21 = getTransformFromPoses(gtT01, gtRp01, gtT02, gtRp02)
    estResults = evaluateEstimatesReTransform(gtRp21, gtT21, R21s, T21s)
    return (gtRp21, gtT21) + estResults


def evaluateEstimatesReTransform(gtR21, gtT21, R21s, T21s):

    # compute error of the result
    # pdb.set_trace()
    quat2To1 = rotMatrixToQuaternion(gtR21)
    dist2To1 = np.linalg.norm(gtT21)

    # Be charitable, take the best hypothesis as the right one
    bestTLenErrorIdx, bestTCosErrorIdx, bestRotErrorIdx = -1, -1, -1
    bestTLenError, bestTCosError, bestRotError = np.inf, np.inf, np.inf
    leastError = np.inf
    winnerIdx = -1
    for i, (hypR, hypT) in enumerate(zip(R21s, T21s)):
        hypT = hypT.flatten()
        TError = gtT21 - hypT
        TLenError = (np.linalg.norm(TError) /
                     max(dist2To1, np.linalg.norm(hypT)))
        if TLenError < bestTLenError:
            bestTLenErrorIdx = i

        TCosError = distance.cosine(hypT, gtT21)
        if TCosError < bestTCosError:
            bestTCosErrorIdx = i

        quatHyp = rotMatrixToQuaternion(hypR)
        rotError = distance.cosine(quat2To1, quatHyp)
        if rotError < bestRotError:
            bestRotErrorIdx = i

        totalError = TLenError + TCosError + rotError
        if totalError < leastError:
            bestTLenError = TLenError
            bestTCosError = TCosError
            bestRotError = rotError
            leastError = totalError
            winnerIdx = i

    # save the results of this pair
    # if winnerIdx == -1:
    #     print "No transform could be found!"
    if len(R21s) > 1:
        print("Of", len(R21s), "returned transforms", winnerIdx,
              "is the best.")
    rotDist2To1 = distance.cosine(quat2To1, [0, 0, 0, 1])
    stats = (dist2To1, bestTLenError, bestTCosError, rotDist2To1, bestRotError)
    return winnerIdx, stats


def getRotationMagnitude(rotMat):
    rotQuat = rotMatrixToQuaternion(rotMat)
    return distance.cosine(rotQuat, [0, 0, 0, 1])
