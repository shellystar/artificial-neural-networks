#evaluate
#TP: positive examples predicted to be positive
#FP: negtive examples predicted to be positive
#TN: negtive examples predicte to be negtive
#FN: positive examples predicted to be negtive

def Evaluate(prediction, test_labels):
    TP = FP = FN = TN = 0
    for i in range(len(test_labels)):
        if test_labels[i] == prediction[i] and test_labels[i] == 1:
            TP += 1
        elif test_labels[i] == prediction[i] and test_labels[i] == 0:
            TN += 1
        elif test_labels[i] != prediction[i] and test_labels[i] == 1:
            FN += 1
        elif test_labels[i] != prediction[i] and test_labels[i] == 0:
            FP += 1

    print TP
    print TN
    print FP
    print FN
    P = 1.0 * TP / (TP + FP)
    print("P", P)
    R = 1.0 * TP / (TP + FN)
    print("R", R)
    F1 = 2 * P * R / (P + R)
    print("F1",F1)
    return F1
