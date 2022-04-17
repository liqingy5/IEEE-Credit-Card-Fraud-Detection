import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve
from ipdb import set_trace

#extract components of the confusion matrix
def conf_matrix(y_test,y_pred_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print([[tp,tn],[fp,fn]])
    print("Accuracy: ",accuracy)
    print("Misclassification error = ",fp+fn)   
    print("SENS(recall)  = ",recall) 
    print("SPEC   = ",tn/(tn+fp)) 
    print("PPV(Precision)   = ",precision) 
    print("NPV   = ",tn/(tn+fn)) 
    print("F1-SCORE = ", 2 * (precision * recall) / (precision + recall))

    

### Roc curve and PR curve
def roc_pr_curve(y_test,probs_predict):
    # calculate AUROC
    Auroc_score=roc_auc_score(y_test, probs_predict)
    print("AUROC score: ",Auroc_score)
    # Draw the ROC curve
    plt.figure(1)
    # ROC curve components
    fpr, tpr, thresholdsROC = roc_curve(y_test, probs_predict)
    #plot
    plt.plot(fpr,tpr)
    plt.title("ROC curve")
    plt.xlabel("1-SPEC")
    plt.ylabel("SENS")
    plt.show
    
    # Draw the PR curve
    plt.figure(2)
    # Components of the Precision recall curvey
    precision, recall, thresholdsPR = precision_recall_curve(y_test, probs_predict)
    # plot
    plt.plot(recall,precision)
    plt.title("PR curve")
    plt.xlabel("SENS (Recall)")
    plt.ylabel("PPV (Precision)")
    plt.show





