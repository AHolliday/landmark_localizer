import cv2
import numpy as np

from . import geometryUtils as geo


def getDoubleImage(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # draw feature points on each image
    view = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    view[:h1, :w1, :] = img1
    view[:h2, w1:, :] = img2
    return view


def scaleImage(image, newMinShape):
    scaleFactor = min(image.shape[:2]) / float(newMinShape)
    newSize = (np.array(image.shape) / scaleFactor).astype(int)
    newSizeCv = (newSize[1], newSize[0])
    return cv2.resize(image, newSizeCv)


def showNow(image, name='showNow', time=0, raiseOnEsc=False):
    cv2.imshow(name, image)
    k = cv2.waitKey(time)
    if raiseOnEsc and k % 27 == 0:
        raise Exception("User pressed escape while displaying!")
    return k


def getRandomColour():
    """
    A useful function for generating colours to display.
    """
    # we want value and saturation above a certain threshold; hue may vary.
    hueMax = 179
    satMax = 255
    valMax = 255
    hue = int(np.random.random()*hueMax)
    satThreshold = satMax/2
    saturation = satThreshold + int(np.random.random()*(satMax - satThreshold))
    valThreshold = valMax/2
    value = valThreshold + int(np.random.random()*(valMax - valThreshold))
    hsvColour = np.uint8([[[hue, saturation, value]]])
    bgrColour = cv2.cvtColor(hsvColour, cv2.COLOR_HSV2BGR)
    return tuple(map(int, bgrColour.flatten()))


def renderHomography(image1, image2, H, showBothOrig=True):
    homogImg = cv2.warpPerspective(image1, H, (image2.shape[1],
                                               image2.shape[0]))
    homogAndOrig = getDoubleImage(homogImg, image2)
    if showBothOrig:
        homogAndOrig = getDoubleImage(image1, homogAndOrig)
    return homogAndOrig


def drawOtherCamera(image, Rp12, T12, K, color=(0, 255, 0), thickness=1):
    camCorners = np.ones((5, 4), dtype=float)
    camCorners[0, :3] = (0, 0, 0)
    xLeft = - K[0, 2] / K[0, 0]
    yTop = - K[1, 2] / K[1, 1]
    xRight = (image.shape[1] - K[0, 2]) / K[0, 0]
    yBottom = (image.shape[0] - K[1, 2]) / K[1, 1]
    camCorners[1, :2] = (xLeft, yTop)
    camCorners[2, :2] = (xLeft, yBottom)
    camCorners[3, :2] = (xRight, yBottom)
    camCorners[4, :2] = (xRight, yTop)

    # compute projection matrix
    Rc21 = Rp12
    P1 = geo.buildProjectionMatrix(K, Rc21, T12)
    # project camCorners into cam 1's frame - 4xN to multiply with projection
    # matrix
    camPts3d = np.dot(P1, camCorners.T)
    camPts3dNorm = camPts3d / camPts3d[2, :]
    camPts = [tuple(row.astype(int)) for row in camPts3dNorm.T[:, :2]]
    camCenter = camPts[0]
    for i in range(1, 5):
        corner = camPts[i]
        # draw lines from the center to the corners
        image = cv2.line(image, camCenter, corner, color, thickness)
        # draw lines between the corners
        if i < 4:
            nextCorner = camPts[i+1]
        else:
            nextCorner = camPts[1]
        image = cv2.line(image, corner, nextCorner, color, thickness)
    # this is just for diagnostics
    # for i in range(5):
    #     image = cv2.putText(image, str(i), camPts[i], cv2.FONT_HERSHEY_SIMPLEX,
    #                         1, (0, 0, 255), thickness=5)
    return image


def drawSeen(image, depth, Rp12, T12, K, color=(0, 255, 0), thickness=1):
    camCorners = np.ones((5, 4), dtype=float)
    camCorners[0, :3] = (0, 0, 0)
    xLeft = - K[0, 2] / K[0, 0]
    yTop = - K[1, 2] / K[1, 1]
    xRight = (image.shape[1] - K[0, 2]) / K[0, 0]
    yBottom = (image.shape[0] - K[1, 2]) / K[1, 1]
    camCorners[1, :2] = (xLeft, yTop)
    camCorners[2, :2] = (xLeft, yBottom)
    camCorners[3, :2] = (xRight, yBottom)
    camCorners[4, :2] = (xRight, yTop)

    # compute projection matrix
    Rc21 = Rp12
    Mc21 = np.eye(4)
    Mc21[:3, :3] = Rc21
    Mc21[:3, 3] = T12.flatten()

    # project camCorners into cam 1's frame - 4xN to multiply with projection
    # matrix
    camPts3d = np.dot(Mc21[:3, :], camCorners.T).T

    # Draw rays down to the plane.
    rays = camPts3d[1:] - camPts3d[0]
    zCenter = camPts3d[0][2]
    # if any rays have zero z-component, they point off to the horizon.
    tIntersects = (depth - zCenter) / rays[:, 2]
    pIntersects = camPts3d[0] + (tIntersects * rays.T).T
    pIntersectsImgNonNorm = np.dot(K, pIntersects.T)
    pIntersectsImg = pIntersectsImgNonNorm / pIntersectsImgNonNorm[2, :]
    planePts = [tuple(row.astype(int)) for row in pIntersectsImg.T[:, :2]]
    for i in range(4):
        corner = planePts[i]
        if i < 3:
            nextCorner = planePts[i+1]
        else:
            nextCorner = planePts[0]
        try:
            cv2.line(image, corner, nextCorner, color, thickness)
        except OverflowError:
            continue

    centerPtNonNorm = np.dot(K, camPts3d[0])
    centerPt = centerPtNonNorm[:2] / centerPtNonNorm[2]
    centerPt = tuple(centerPt.astype(int))
    try:
        cv2.circle(image, centerPt, thickness, color, thickness=thickness)
    except OverflowError:
        pass

    # this is just for diagnostics
    # for i in range(5):
    #     cv2.putText(image, str(i), camPts[i], cv2.FONT_HERSHEY_SIMPLEX,
    #                         1, (0, 0, 255), thickness=5)
    return image


# Visualization tools

def renderBoxMatches(image1, boxes1, image2, boxes2, matches,
                     showOnlyMatchedBoxes=True, renderBackground=True,
                     renderLines=False, thickness=2):
    if len(matches) == 0:
        # nothing to render!
        doubleImage = getDoubleImage(image1, image2)
        cv2.putText(doubleImage, "No matches found!",
                    (doubleImage.shape[1] / 2, doubleImage.shape[0] / 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), thickness=thickness)
        return doubleImage

    images = [image1, image2]
    bboxLists = [boxes1, boxes2]
    colours = [getRandomColour() for _ in matches]
    if showOnlyMatchedBoxes:
        matches1, matches2 = list(zip(*matches))
        matchedBoxes1 = np.array(boxes1)[matches1, :]
        matchedBoxes2 = np.array(boxes2)[matches2, :]
        mbLists = [matchedBoxes1, matchedBoxes2]
        dispImage1, dispImage2 = [renderBoxes(i, b, thickness=thickness, colours=colours)
                                  for i, b in zip(images, mbLists)]
    else:
        dispImage1, dispImage2 = [renderBoxes(i, b, thickness=thickness)
                                  for i, b in zip(images, bboxLists)]

    # render the boxes on top of each image
    dispImage = getDoubleImage(dispImage1, dispImage2)

    if renderLines:
        # render lines between matching boxes
        for (i1, i2), colour in zip(matches, colours):
            box1Center = geo.getCenterFromBox(boxes1[i1])
            box2Center = geo.getCenterFromBox(boxes2[i2])
            box2Center[0] += dispImage1.shape[1]
            if renderBackground:
                cv2.line(dispImage, tuple(box1Center.astype(int)),
                         tuple(box2Center.astype(int)), (0, 0, 0),
                         thickness=thickness)
            cv2.line(dispImage, tuple(box1Center.astype(int)),
                     tuple(box2Center.astype(int)), colour, thickness=thickness)
    return dispImage


def renderBoxes(image, boxes, showNumBoxes=None, drawCenters=False,
                colour=None, colours=None, thickness=1):
    if type(image) is str:
        image = cv2.imread(image)
    dispImage = image.copy()

    if type(boxes) is not np.ndarray:
        boxes = np.array(boxes)

    # show all boxes at once
    if showNumBoxes is not None and len(boxes) > showNumBoxes:
        boxes = boxes[:showNumBoxes]
    if not colours:
        if colour:
            colours = [colour] * len(boxes)
        else:
            colours = [getRandomColour() for _ in boxes]
    for boxRow, c in zip(boxes.astype(int), colours):

        # there may be a fifth column representing confidence
        x1, y1, x2, y2 = boxRow[:4]
        # y comes before x in opencv axis ordering
        if drawCenters:
            cv2.circle(dispImage, ((x1+x2/2), (y1+y2/2)), 1, c,
                       thickness)
        else:
            cv2.rectangle(dispImage, (x1, y1), (x2, y2), c, thickness)

    return dispImage
