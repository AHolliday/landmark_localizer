import os
import pdb
import cv2
import argparse
import logUtils
import pickle
import numpy as np

IMG_COLUMN = "leftImage"
LAT_COLUMN = "Latitude"
LON_COLUMN = "Longitude"

# should be red
POS_COLOUR = (0,0,255)
M_PER_DEG_LAT = 110560
# just an estimate
VIEW_VERT_DIST_M = 5.36

def getPos(logline, latIndex, lonIndex):
    fields = logline.split(logUtils.LOG_ENTRY_SEP)
    return np.array([float(fields[i]) for i in [latIndex, lonIndex]])


def findClosestLog(searchPos, targetLogs, latIndex, lonIndex, maxDist=None):
    minDist = None
    nearestLog = None
    for log in targetLogs:
        # get position
        targetPos = getPos(log, latIndex, lonIndex)
        distance = np.linalg.norm(targetPos - searchPos) * M_PER_DEG_LAT
        if maxDist is not None and distance > maxDist:
            # if it's outside the radius, continue
            continue
        elif minDist is None or distance < minDist:
            minDist = distance
            nearestLog = log
    return nearestLog


def loadAndRenderMatch(query, match, showQueryRegion=True, dispWidth=None):
    # load the two images
    qImg = cv2.imread(query[0])
    matchImg = cv2.imread(match[0])

    qToMVec = query[1] - match[1]
    qToMVec_m = qToMVec * M_PER_DEG_LAT
    dist_m = np.linalg.norm(qToMVec_m)

    if showQueryRegion:
        # render a circle with the circle on which the query should roughly lie 
        if dist_m < VIEW_VERT_DIST_M:
            imgCenter = tuple(np.array(matchImg.shape[:2]) / 2)
            cvImgCenter = (imgCenter[1], imgCenter[0])
            dist_pix = int(dist_m * imgCenter[0] / float(VIEW_VERT_DIST_M))
            cv2.circle(matchImg, cvImgCenter, dist_pix, POS_COLOUR)
    
    # compute distance between the two
    displayName = "dist_m: " + str(dist_m)
    # dispImg = renderMatch(qImg, matchImg, displayName)
    maxShape = (max(qImg.shape[0], matchImg.shape[0]),
                max(qImg.shape[1], matchImg.shape[1]))
    imgContainer = np.zeros((maxShape[0], maxShape[1]*2, 3), dtype='uint8')
    # put the source image in the left half
    imgContainer[:qImg.shape[0], :qImg.shape[1], :] = qImg
    # put the closest image in the right half
    imgContainer[:matchImg.shape[0],
                 maxShape[1]:maxShape[1] + matchImg.shape[1], :] = matchImg
    if dispWidth:
        dispHeight = 2 * int(dispWidth * (maxShape[1] / float(maxShape[0])))
        imgContainer = cv2.resize(imgContainer, (dispHeight, dispWidth))
    return imgContainer, displayName


def showMatches(matches):
    for query, match in matches:
        dispImg, displayName = loadAndRenderMatch(query, match)
        cv2.imshow(displayName, dispImg)
        key = cv2.waitKey(0)
        if key == 27:
            print "Escape key pressed!  Exiting."
            exit(0)
        cv2.destroyAllWindows()


def writeMatches(matches, outdir):
    for query, match in matches:
        dispImg, displayName = loadAndRenderMatch(query, match)
        qFilename = os.path.basename(query[0])
        filename = os.path.join(outdir, 'result_' + qFilename)
        cv2.imwrite(filename, dispImg)
    
    
def buildClosestList(queryLogFilename, targetLogFilename):
    qLogFile = open(queryLogFilename, 'r')
    tLogFile = open(targetLogFilename, 'r')

    # get indices of desired fields
    header = qLogFile.readline()
    iImg, iLat, iLon = (logUtils.getColumnIndex(header, c)
                        for c in [IMG_COLUMN, LAT_COLUMN, LON_COLUMN])
    # discard header from target log file (assume it's the same as query log)
    tLogFile.readline()

    # read the two logs into memory
    qLogs = [qLog for qLog in qLogFile]
    tLogs = [tLog for tLog in tLogFile]

    # for each query log, find the closest target log (brute force)
    matches = []
    for qLog in qLogs:
        qPos = getPos(qLog, iLat, iLon)
        qImage = qLog.split(logUtils.LOG_ENTRY_SEP)[iImg]
        closestLog = findClosestLog(qPos, tLogs, iLat, iLon, VIEW_VERT_DIST_M)
        if not closestLog:
            continue
        closestImage = closestLog.split(logUtils.LOG_ENTRY_SEP)[iImg]
        closestPos = getPos(closestLog, iLat, iLon)
        entry = ((qImage, qPos, qLog), (closestImage, closestPos, closestLog))
        matches.append(entry)

    print "Of", len(qLogs), "queries,", len(matches), "matches found"
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--matches", help="\
    Use the matches in the provided pickle file instead of computing them")
    parser.add_argument("-q", "--query", help="\
    The log file containing images to serve as queries")
    parser.add_argument("-t", "--target", help="\
    The log file containing images to match with queries")
    parser.add_argument("-s", "--silent", action="store_true", help="\
    silent: don't display results, just compute and save the matches")
    parser.add_argument("--save", help="\
    save pairs of images to the provided directory")
    args = parser.parse_args()

    matches = []
    if args.query and args.target:
        # compute the matches
        matches = buildClosestList(args.query, args.target)
        pklFilename = None
        if args.matches:
            pklFilename = args.matches
        else:
            pklFilename = 'matches.pkl'
        with open(pklFilename, 'w') as pklFile:
            pickle.dump(matches, pklFile)
        
    elif args.matches:
        # load pre-computed matches
        with open(args.matches, 'r') as pklFile:
            matches = pickle.load(pklFile)
    else:
        raise ValueError("\
        Must provide both a query and a target logfile as arguments, or else \
        provide a file containing precomputed matches.")

    if not args.silent:
        showMatches(matches)
    if args.save:
        writeMatches(matches, args.save)
    
