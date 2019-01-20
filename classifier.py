import csv
from structure import NewsGroup
import numpy as np
import math

data  = [np.empty([3000,61188]) for x in range(20)]
temp = np.empty([1,61188])
c = [0 for x in range(20)]
location = "data/training.csv"
testingLocation = "data/testing.csv"
length = 61189
v = length -1
b = 1/v
a = 1+b
print ("done")

with open(location, 'r') as f:
    reader = csv.reader(f)
    count = 1
    for row in reader:
        label = int(row[length])-1
        newData = row[1:length]
        data[label][c[label]] = newData
        c[label]+=1
        count = count+1
        if (count % 500 == 0):
            print("looping")
    print ("done")
f.close()

print ("converting to numpy")
data = [data[x][0:(c[x]+1)] for x in range(20)]

print ("done")

total = [x.shape[0] for x in data]
totalWords = [np.sum(x) for x in data]
xCounts = [np.sum(x,axis=0) for x in data]
py = total/np.sum(total)
pxy = [(x[0]+b)/(x[1]+(b*length)) for x in zip(xCounts,totalWords)]
pxy = np.log2(pxy)
py = np.log2(py)
pxyt = pxy.transpose()

def predict(newdata):
    e1 = np.matmul(newdata,pxyt)
    e2 = np.add(py,e1)
    return np.argmax(e2)+1

print ("predicting")

with open(testingLocation,'r') as csvFileObject:
    prediction = [['id','class']]
    csvReader = csv.reader(csvFileObject)
    count = 0
    for line in csvReader:
        temp[0] = line[1:length]
        prediction.append([line[0],predict(temp)])
    print ("writing")
    predictionFile = open('predict.csv','w')
    with predictionFile:
        writer = csv.writer(predictionFile)
        writer.writerows(prediction)
