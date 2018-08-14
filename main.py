import tensorflow as tf
import numpy as np
import DataOperation
import ANN
import Evaluation

train_labels, train_features, test_labels, test_features = DataOperation.readData()
Wi, Wo = ANN.BPTrain(train_features, train_labels, 9, 4, 1,len(train_labels))
#print(Wi)
#print(Wo)
prediction = ANN.predict(test_features, test_labels, Wi, Wo)
coun1 = 0
coun2 = 0
#print(prediction)
for l in prediction:
    if l == 1:
        coun1 += 1
    else:
        coun2 += 1
print("count positive",coun1)
print("count negtive",coun2)
F1 = Evaluation.Evaluate(prediction, test_labels)