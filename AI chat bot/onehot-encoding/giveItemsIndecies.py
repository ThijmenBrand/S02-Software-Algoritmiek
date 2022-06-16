from alphabaticSort import sortAlphabetically


def assignValuesToItems(inputArr):
    sortedArr = sortAlphabetically(inputArr.copy())
    unique = []
    output = []
    returnArr = []
    for inputItem in sortedArr:
        if inputItem not in unique:
            unique.append(inputItem)
    for i in unique:
        output.append([i, len(output)])
    for p in inputArr:
        for l in output:
            if p == l[0]:
                returnArr.append(l[1])
    return returnArr