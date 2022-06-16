def findUniqueItems(inputArr):
    #Array to keep track of unique items
    unique = []
    #Array to place the refined data in
    output = []
    #For every item in inputArr. This is the item and is equivalent to for(int i = 0; i <= inputArr.count; i++) {inputArr[i]}
    for inputItem in inputArr:
        #If the item in input array is not yet in the unique array (So it is still unique), place it there
        if inputItem not in unique:
            unique.append(inputItem)
    # for every item in the unique array
    for i in unique:
        #counter to count how many times an item occurs
        counter = 0
        #for every item in the input array
        for j in inputArr:
            #check if the item in input array is equal to the item in unique arr.
            if j == i:
                #If it is equal (So the item matches) the counter goes up
                counter += 1
        output.append([i, counter])

    return output

print(findUniqueItems([1,1,2,3,4,5,6,6,7,8,9,10]))