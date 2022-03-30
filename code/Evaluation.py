import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve
import pickle
import metrics

with open('.env') as f:
    _data_pth_ = f.readline().strip()
    _data_pth_ = os.path.expanduser(_data_pth_)


def evaluate(model_name):
    data = pd.read_csv(f'{_data_pth_}/processed/train_transaction.csv', index_col=0)
    y, X = data['isFraud'], data.drop(columns=['isFraud'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    _filename_ = model_name+".model"
    # Load the Model back from file
    with open(f'../models/{_filename_}', 'rb') as file:  
        model = pickle.load(file)
        
    if model_name =="XGboost":
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        y_pred_probs = model.predict(dtest)
        metrics.roc_pr_curve(y_test,y_pred_probs)
        y_pred_probs[y_pred_probs >= 0.5] = 1
        y_pred_probs[y_pred_probs < 0.5] = 0
        metrics.conf_matrix(y_test,y_pred_probs)
        return

    
    sc_train = model.score(X_train, y_train)
    sc_test = model.score(X_test, y_test)
    y_pred_test = model.predict(X_test)
    probs=model.predict_proba(X_test)
    print(sc_train)
    print(sc_test)

    metrics.conf_matrix(y_test,y_pred_test)
    metrics.roc_pr_curve(y_test,probs[:,1])
    



