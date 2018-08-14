import csv
import random
import numpy as np

#read data from .csv
def readData():
    rf = open('HTRU_2.csv','r')
    dataReader = csv.reader(rf)
    posDataList = []
    negDataList = []
    featureIndex = 8
    featureMin = [999,999,999,999,999,999,999,999]
    featureMax = [0,0,0,0,0,0,0,0]
    for rList in dataReader:
        for i in range(featureIndex):
            if float(rList[i]) > float(featureMax[i]):
                featureMax[i] = float(rList[i])
            if float(rList[i]) < float(featureMin[i]):
                featureMin[i] = float(rList[i])
        if rList[featureIndex] == '0':
            negDataList.append(rList)
        else:
            posDataList.append(rList)
    rf.close()
    #   17,898 total examples.
    #	1,639 positive examples.
    #	16,259 negative examples.

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    negDataList = random.sample(negDataList, len(posDataList))
    #0.1,0.5 92%
    posTestList = random.sample(posDataList, int(round(len(posDataList) * 0.2)))
    negTestList = random.sample(negDataList, int(round(len(negDataList) * 0.2)))

    #devide postive examples into test and train
    for pTest in posTestList:
        fList = []
        for i in range(featureIndex):
            fList.append((float(pTest[i])-featureMin[i])/(featureMax[i]-featureMin[i]))
        test_labels.append(1)
        fList.append(1)
        test_features.append(fList)

        posDataList.remove(pTest)

    for pTrain in posDataList:
        fList = []
        for i in range(featureIndex):
            fList.append((float(pTrain[i])-featureMin[i])/(featureMax[i]-featureMin[i]))
            train_labels.append(1)
        fList.append(1)
        train_features.append(fList)


    # devide negtive examples into test and train
    for nTest in negTestList:
        fList = []
        for i in range(featureIndex):
            fList.append((float(nTest[i])-featureMin[i])/(featureMax[i]-featureMin[i]))
        test_labels.append(0)
        fList.append(1)
        test_features.append(fList)

        negDataList.remove(nTest)

    for nTrain in negDataList:
        fList = []
        for i in range(featureIndex):
            fList.append((float(nTrain[i])-featureMin[i])/(featureMax[i]-featureMin[i]))
        train_labels.append(0)
        fList.append(1)
        train_features.append(fList)

    train_labels = np.array(train_labels)
    train_features = np.array(train_features)
    test_labels = np.array(test_labels)
    test_features = np.array(test_features)
    return train_labels, train_features, test_labels, test_features

