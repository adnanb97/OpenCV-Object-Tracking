NUMBER_OF_ITERATIONS = 15
tuplesStr = [[] for i in range(NUMBER_OF_ITERATIONS) ]
tuplesFlt = [[] for i in range(NUMBER_OF_ITERATIONS) ]
counter = 0;
algoritam = 'csrt'
filename = 'Podaci/BoundingBox01' + algoritam + '.txt'

#read a file
with open(filename, 'r') as f:
    lines = f.readlines()


for line in lines: 
    # if we get to a substring iteration
    if (line[0:9] == "Iteration"):
        counter = counter + 1
    else: 
        tuplesStr[counter - 1].append(line.strip('\n'))


counter = 0
for tupleStr in tuplesStr: 
    
    if (tupleStr != []):
        counter = counter + 1
        for elementTupleStr in tupleStr: 
            row = elementTupleStr.split(',')
            tupleElements = row[0].split(' ')
            separatedRow = (float(tupleElements[0]), float(tupleElements[1]), float(tupleElements[2]), float(tupleElements[3]))
            #print (str(counter) + str(separatedRow))

            tuplesFlt[counter - 1].append(separatedRow)

size = len(tuplesFlt) #how many iterations
sizeOfIteration = []
for tupleFlt in tuplesFlt: 
    sizeOfIteration.append(len(tupleFlt)) 

vel = min(sizeOfIteration) #size of each iteration

#empty arrays for averaging frames

bestPathForEachFrame0 = [0] * vel 
bestPathForEachFrame1 = [0] * vel 
bestPathForEachFrame2 = [0] * vel 
bestPathForEachFrame3 = [0] * vel 


#print (tuplesFlt)
for tupleFlt in tuplesFlt:
    for i in range(vel): 
        bestPathForEachFrame0[i] += tupleFlt[i][0]
        bestPathForEachFrame1[i] += tupleFlt[i][1]
        bestPathForEachFrame2[i] += tupleFlt[i][2]
        bestPathForEachFrame3[i] += tupleFlt[i][3]
imeIdealnePutanje = 'idealna'
filename = 'IdealnaPutanja/' + imeIdealnePutanje + '.txt'
character = 'w'
file = open(filename, character) 

for i in range(vel): 
    thres = 0
    a1 = bestPathForEachFrame0[i] / (float(size) + thres)
    a2 = bestPathForEachFrame1[i] / (float(size) + thres)
    a3 = bestPathForEachFrame2[i] / (float(size) + thres)
    a4 = bestPathForEachFrame3[i] / (float(size) + thres)
    
    file.write(str(a1) + ' ' + str(a2) + ' ' + str(a3) + ' ' + str(a4) + "\n")       

print("All data written to " + imeIdealnePutanje)