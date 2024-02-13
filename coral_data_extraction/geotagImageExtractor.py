#!/usr/bin/env python

CREATION_DATE_KEY = 'creation_date'
DURATION_KEY = 'duration'
METADATA_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_TIME_FORMAT = '%H:%M:%S.%f %m/%d/%Y'
IMG_FILENAME_FORMAT = '%Y-%m-%d_%H-%M-%S-%f'
LOG_ENTRY_SEP = ';'
OUT_LOG_FILENAME = 'log'

import pdb
import argparse
import os
import cv2
from datetime import datetime, timedelta
from hachoir_core.cmd_line import unicodeFilename
from hachoir_parser import createParser
from hachoir_metadata import extractMetadata


class VidListFrameIterator(object):
    
    def __init__(self, vidList):
        self.vidListIter = iter(vidList)
        self.nextVid()
        self.startTime = self.currVidStartTime

    def nextVid(self):
        try:
            self.currVid.release()
        except:
            pass
        nextVidName = self.vidListIter.next()
        self.currVidStartTime = getVidStartTime(nextVidName)
        self.currVid = cv2.VideoCapture(nextVidName)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        isNotDone, nextFrame = self.currVid.read()
        # if the video is over, move to the next one in the list.
        while not isNotDone:
            # this will raise StopException after the last vid
            self.nextVid()
            isNotDone, nextFrame = self.currVid.read()

        # get the timestamp of the frame
        currTime_ms = self.currVid.get(cv2.CAP_PROP_POS_MSEC)
        currTime = timedelta(milliseconds = currTime_ms)
        frameTime = self.currVidStartTime + currTime
        return nextFrame, frameTime
        

def getVidStartTime(vidFilename):
    parser = createParser(unicodeFilename(vidFilename), vidFilename)
    metadata = extractMetadata(parser)
    timeText = metadata.getText(CREATION_DATE_KEY)
    return datetime.strptime(timeText, METADATA_TIME_FORMAT)

    # fps = vidCap.get(cv2.CAP_PROP_FPS)
    # numFrames = vidCap.get(cv2.CAP_PROP_FRAME_COUNT)
    # CAP_PROP_POS_MSEC


def getLogEntryTime(splitLogEntry):
    timeText = splitLogEntry[2]
    dateText = splitLogEntry[3]
    datetimeText = timeText + ' ' + dateText
    return datetime.strptime(datetimeText, LOG_TIME_FORMAT)


def getClosestFrame(frameIter, targetTime):
    lastFrame, lastTime = None, None
    writeFrame, frameTime = None, None
    for frame, time in frameIter:
        if targetTime == time:
            writeFrame, frameTime = frame, time
            break
        else:
            if lastFrame is not None and lastTime is not None:
                # check if this timestamp is between the last and current frames
                if lastTime < targetTime and targetTime < time:
                    # write the closer frame
                    deltaNext = abs(targetTime - time)
                    deltaLast = abs(targetTime - lastTime)
                    if deltaLast < deltaNext:
                        writeFrame, frameTime = lastFrame, lastTime
                    else:
                        writeFrame, frameTime = frame, time
                    break
                
            lastFrame, lastTime = frame, time
    return writeFrame, frameTime


def writeFrame(frame, time, outDirName, prefix=''):
    imgFilename = prefix + time.strftime(IMG_FILENAME_FORMAT) + '.png'
    imgPath = os.path.join(outDirName, imgFilename)
    cv2.imwrite(imgPath, frame)
    return imgPath


def produceTimeStampedImages(logFilename, leftVidList, outLogFilename,
                             outDirName, filePrefix='', rightVidList=None):
    outlog = open(os.path.join(outDirName, outLogFilename), 'w')
    inlog = open(logFilename, 'r')
    leftFrameIter = VidListFrameIterator(leftVidList)
    rightFrameIter = None
    if rightVidList is not None and len(rightVidList) == len(leftVidList):
        rightFrameIter = VidListFrameIterator(rightVidList)
    # TODO implement stereo

    # iterate over the log entries
    for logEntry in inlog:
        values = logEntry.split(LOG_ENTRY_SEP)

        # skip the entry if the first value is not a number
        try:
            float(values[0])
        except:
            header = ''
            if rightFrameIter is None:
                header = LOG_ENTRY_SEP.join(['image', logEntry])
            else:
                header = LOG_ENTRY_SEP.join(['leftImage', 'rightImage', 
                                             logEntry])
            outlog.write(header)
            continue
        
        # start at first entry in log file that's older than vidStartTime
        entryTime = getLogEntryTime(values)
        if entryTime < leftFrameIter.startTime:
            continue

        frame, time = getClosestFrame(leftFrameIter, entryTime)
        if frame is None:
            # we've reached the end of the videos
            break
        # write frame to image file
        if rightFrameIter is None:
            imgFilename = writeFrame(frame, time, outDirName, filePrefix)
            outLogEntry = imgFilename + LOG_ENTRY_SEP + logEntry
        else:
            lPrefix = 'L_' + filePrefix
            lFilename = writeFrame(frame, time, outDirName, lPrefix)
            rightFrame, rightTime = getClosestFrame(rightFrameIter, entryTime)
            rPrefix = 'R_' + filePrefix
            rFilename = writeFrame(rightFrame, rightTime, outDirName, rPrefix)
            outLogEntry = LOG_ENTRY_SEP.join([lFilename, rFilename, logEntry])
            
        # write its info (pos etc.) to the tagging file
        outlog.write(outLogEntry)

    outlog.close()
    inlog.close()


def strLinesToList(contentsStr):
    vidList = contentsStr.split('\n')
    # strip whitespace characters
    vidList = map(lambda x: x.strip(), vidList)
    # remove empty lines
    vidList = filter(lambda x: x is not '', vidList)
    return vidList


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    parser.add_argument('vidListFile')
    parser.add_argument('outDir')
    parser.add_argument('-p', '--prefix', default='')
    parser.add_argument('-r', '--rightVidListFile')
    args = parser.parse_args()
    
    with open(args.vidListFile, 'r') as f:
        vidList = strLinesToList(f.read())

    rightVidList = None
    if args.rightVidListFile:
        with open(args.rightVidListFile, 'r') as f:
            rightVidList = strLinesToList(f.read())
            
    produceTimeStampedImages(args.log, vidList, OUT_LOG_FILENAME, 
                             args.outDir, args.prefix, rightVidList)
