import argparse
import logUtils

DESIRED_COLUMNS = ["leftImage",
                   "Latitude",
                   "Longitude",
                   "DTB Height (m)",
                   "Roll Angle",
                   "Pitch Angle",
                   "C Magnetic Heading"
]


def getColumns(inFilename, outFilename, columns):
    fi = open(inFilename, 'r')
    fo = open(outFilename, 'w')

    # read in header
    header = fi.readline()
    # get indices of desired columns (img, x, y, z, r, p, y)
    columnIndices = [logUtils.getColumnIndex(header, column)
                     for column in columns]
    for inLine in fi:
        # get the values at the desired indices
        inFields = inLine.strip().split(logUtils.LOG_ENTRY_SEP)
        # write the values at the desired indices to the output file
        outFields = [inFields[i] for i in columnIndices]
        outLine = logUtils.LOG_ENTRY_SEP.join(outFields) + '\n'
        fo.write(outLine)
    
    fi.close()
    fo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()
    getColumns(args.infile, args.outfile, DESIRED_COLUMNS)
