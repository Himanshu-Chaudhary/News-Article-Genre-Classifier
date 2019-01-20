import numpy as np
import csv
import random
from scipy.sparse import csr_matrix

class Logres:
    def __init__(self):
        self.length = 61189
        self.w = np.zeros([20,61189])
        self.data  = np.zeros([15000,61189])
        self.n = 0.01
        self.l = 0.01

    def readData(self,location):
        data  = np.zeros([15000,61189])
        length = self.length
        c = 0
        with open(location, 'r') as f:
            reader = csv.reader(f)
            count = 1
            for row in reader:
                newData = row[1:length+1]
                data[c] = newData
                c+=1
                count = count+1
                if (count % 1000 == 0):
                    break
                    print("looping")
            print ("done")
        f.close()
        data = data[0:(c+1)]
        return data

    def setData(self,data):
        length = self.length
        self.realY = data[:,length-1]
        data = np.delete(data,length-1,axis=1)
        data = np.hstack((np.ones((data.shape[0],1)),data))
        self.y = np.array([[1 if (x1==x2+1) else 0 for x1 in self.realY] for x2 in range(20)])
        self.data = data

    def logres(self):
        w = self.w
        s = csr_matrix(self.data)
        y = self.y
        l = self.l
        n = self.n

        count = 0
        for i in range (7000):
            #print(w)
            pxy1 = np.exp(np.divide((w*s.transpose()),10000))
            pxy1[:,pxy1.shape[1]-1] = np.ones([1,20])
            pxy1 = pxy1/pxy1.sum(axis=0,keepdims=1)
            diff1 = np.subtract(y,pxy1)
            lw1 = np.multiply(l,w)
            ypx1 = diff1*s
            final = np.subtract(ypx1,lw1)
            w = np.add(w,np.multiply(n,final))
            count = count+1
            if (count % 100 == 0):
                break
                print("looping")
        print ("done")
        self.w = w

    def predict(self,newdata):
        e1 = np.matmul(self.w,newdata.transpose())
        return np.argmax(e1)+1

    def predictTofile(self,location):
        print ("predicting")
        temp = np.empty([1,61189])
        with open(location,'r') as csvFileObject:
            prediction = [['id','class']]
            csvReader = csv.reader(csvFileObject)
            count = 0
            for line in csvReader:
                id = line[0]
                line[0]=1
                temp[0] = line
                prediction.append([id,self.predict(temp)])
            print ("writing")
            predictionFile = open('predict.csv','w')
            with predictionFile:
                writer = csv.writer(predictionFile)
                writer.writerows(prediction)

    def test(self,trainData,l,n,testData):
        self.setData(trainData)
        self.setln(l,n)
        self.logres()
        print ("here")
        self.predictMatrix(testData)

    def setln(self,l,n):
        self.l = l
        self.n = n

    def predictMatrix(self, testData):
        output = []
        y = testData[:,testData.shape[1]-1].shape
        testData = np.delete(testData,testData.shape[0],axis=1)
        testData = np.hstack((np.ones((testData.shape[0],1)),testData))
        temp = np.empty([1,61189])
        for i in range(testData.shape[0]):
            temp[0] = testData[i]
            output.append(self.predict(temp))
        print (output)



def main():
    lg = Logres()
    trainlocation = "data/training.csv"
    testLocation = "data/testing.csv"

    data = lg.readData(trainlocation)
    lg.test(data[0:500],0.1,0.1,data[500:700])


if __name__ == "__main__":
    main()
