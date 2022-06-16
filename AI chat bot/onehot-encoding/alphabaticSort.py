def sortAlphabetically(inputArr):
    for i in range(len(inputArr)):
        j = inputArr[i:].index(min(inputArr[i:])) + i
        inputArr[i], inputArr[j] = inputArr[j], inputArr[i]
    return inputArr