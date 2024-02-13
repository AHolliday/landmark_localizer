import argparse
import os
import shutil
import logUtils

OUT_LOG_FILENAME = 'filtered.log'
DEPTH_COLUMN = "DFS Depth (m)"

def cleanupLogfile(dirname, logfilename):
    # read contents of directory
    files = os.listdir(dirname)

    # open files for reading and writing
    fi = open(logfilename, 'r')
    fo = open(os.path.join(dirname, OUT_LOG_FILENAME), 'w')
    
    # deal with the header first
    header = fi.readline()
    fo.write(header)

    for logEntry in fi:
        # get the left image filename
        fields = logEntry.split(logUtils.LOG_ENTRY_SEP)
        leftPath, rightPath = fields[0], fields[1]
        leftFilename = os.path.basename(leftPath)
        # if the filename is in the dir:
        if leftFilename in files:
            # write the line to the new file
            fo.write(logEntry)
        elif os.path.exists(rightPath):
            # delete the right filename
            os.remove(rightPath)

    # clean up resources
    fi.close()
    fo.close()


def cleanupImages(inDirname, logFilename, outDirname):
    files = os.listdir(inDirname)
    fi = open(os.path.join(inDirname, logFilename), 'r')
    # skip header
    fi.readline()
    fo = open(os.path.join(outDirname, logFilename), 'w')
    for logEntry in fi:
        # get left and right image paths
        fields = logEntry.strip().split(logUtils.LOG_ENTRY_SEP)
        leftPath, rightPath = fields[0], fields[1]
        shutil.copy(leftPath, outDirname)
        shutil.copy(rightPath, outDirname)
        leftFilename = os.path.basename(leftPath)
        rightFilename = os.path.basename(rightPath)
        newLeftPath = os.path.join(outDirname, leftFilename)
        newRightPath = os.path.join(outDirname, rightFilename)
        # copy over this file
        fields[0] = newLeftPath
        fields[1] = newRightPath
        newLogEntry = logUtils.LOG_ENTRY_SEP.join(fields) + '\n'
        fo.write(newLogEntry)

    fi.close()
    fo.close()


def filterByDepth(logfilename, outfilename, minDepth):
    fi = open(logfilename, 'r')
    fo = open(outfilename, 'w')

    header = fi.readline()
    depthIndex = logUtils.getColumnIndex(header, DEPTH_COLUMN)
    fo.write(header)

    for logEntry in fi:
        fields = logEntry.split(logUtils.LOG_ENTRY_SEP)
        depth = float(fields[depthIndex])
        if depth > minDepth:
            fo.write(logEntry)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("-l", "--logfile", default="log")
    parser.add_argument("--depth", type=float, help="\
    If this arg given, filters out images with depth less than given value")
    args = parser.parse_args()

    # get absolute path of provided logfile
    logPath = args.logfile
    if not os.path.isabs(args.logfile):
        logPath = os.path.join(args.dir, logPath)

    if args.depth:
        outFilename = ''.join(["min_depth_", str(args.depth), ".log"])
        outPath = os.path.join(args.dir, outFilename)
        filterByDepth(logPath, outPath, args.depth)
    else:
        cleanupLogfile(args.dir, logPath)
