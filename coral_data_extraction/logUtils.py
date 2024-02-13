LOG_ENTRY_SEP = ';'

def getColumnIndex(header, targetColumnName):
    for i, columnName in enumerate(header.split(LOG_ENTRY_SEP)):
        columnName = columnName.strip()
        if columnName == targetColumnName:
            return i

    raise Exception(' '.join(["\getColumnIndex could not find column", 
                              targetColumnName, "in the header", header]))
