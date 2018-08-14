import numpy as np
import math

def sigmoid(x):
    active = []
    for i in x:
        active.append(1.0 / (1.0 + np.exp(-i)))
    active = np.array(active)
    return active

#f'(x)
def sigmoid_derivate(f):
    return f * (1 - f)

def BPTrain(feature, label, Ni, Nh, No,n):
    Wi = np.random.random((Ni, Nh))
    Wo = np.random.random((Nh, No))
    LearningRate = 0.001
    h = 0.01

    iter = 1
    while(iter > 0):
        for i in range(0, len(feature)):
            #forward
            nethValue = np.dot(feature[i], Wi)
            nethActive = sigmoid(nethValue)
            netoValue = np.dot(nethActive, Wo)
            netoActive = sigmoid(netoValue)

            #back
            #sum of squared errors: 1/2(t-y)^2
            loss_derivate_se = label[i] - netoActive
            #cross entropy
            loss_derivate_ce = np.log(netoActive / (1 - netoActive))
            #print(loss_derivate_ce)
            outputDelta = loss_derivate_ce * sigmoid_derivate(netoActive)
            hiddenDelta = np.dot(Wo, outputDelta) * sigmoid_derivate(nethActive)
            for j in range(0, No):
                Wo[:,j] += LearningRate * (outputDelta[j] * nethActive + h/n * Wo[:,j])
            for k in range(0, Nh):
                Wi[:,k] += LearningRate * (hiddenDelta[k] * feature[i] + h/n * Wi[:,k])

        iter -= 1
        return Wi, Wo

def predict(testFeature, testLabel, Wi, Wo):
    hiddenValue = np.dot(testFeature, Wi)
    hiddenActive = sigmoid(hiddenValue)
    outputValue = np.dot(hiddenActive, Wo)
    outputActive = sigmoid(outputValue)
    prediction = []
    for i in range(0, len(testFeature)):
        if outputActive[i] > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction