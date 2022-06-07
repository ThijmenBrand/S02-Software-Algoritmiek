from giveItemsIndecies import assignValuesToItems
import numpy

def oneHotEncoding(inputArr):
    itemValues = assignValuesToItems(inputArr.copy())
    returnArr = []
    maxVal = max(itemValues)
    binaryArr = numpy.zeros(maxVal+1, dtype=int)
    for i in itemValues:
        binaryCopy = binaryArr.copy()
        binaryCopy[i] = 1 
        returnArr.append(binaryCopy.tolist())
    return returnArr

print(oneHotEncoding(["peter", "peter", "frans", "albert", "coen", "frans"]))
