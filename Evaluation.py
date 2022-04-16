import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import metrics
import utils

with open('.env') as f:
    _data_pth_ = f.readline().strip()
    _data_pth_ = os.path.expanduser(_data_pth_)


def evaluate(model_name, data_name):
    data = pd.read_csv(f'{_data_pth_}/processed/{data_name}.csv', index_col=0)
    y, X = data['isFraud'], data.drop(columns=['isFraud'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state = utils._random_seed_)

    # Load the Model back from file
    with open(f'{utils._data_pth_}/models/{model_name}.model', 'rb') as file:  
        model = pickle.load(file)

        
    if "XGboost" in model_name:
        import xgboost as xgb
        print("Model: XGboost")
        print("Train data")
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        y_pred_probs = model.predict(dtrain)
        metrics.roc_pr_curve(y_train,y_pred_probs)
        y_pred_probs[y_pred_probs >= 0.5] = 1
        y_pred_probs[y_pred_probs < 0.5] = 0
        metrics.conf_matrix(y_train,y_pred_probs)
        print("Test data")
        dtest = xgb.DMatrix(data=X_test, label=y_test)
        y_pred_probs = model.predict(dtest)
        metrics.roc_pr_curve(y_test,y_pred_probs)
        y_pred_probs[y_pred_probs >= 0.5] = 1
        y_pred_probs[y_pred_probs < 0.5] = 0
        metrics.conf_matrix(y_test,y_pred_probs)
        return
    elif "LR" in model_name:
        from sklearn.linear_model import SGDClassifier as SGD
        
        print("Model: Logistic Regression")
        print("Train data")
        y_pred_train = model.predict(X_train)
        probs=model.predict_proba(X_train)
        metrics.conf_matrix(y_train,y_pred_train)
        metrics.roc_pr_curve(y_train,probs[:,1])
        print("Test data")
        y_pred_test = model.predict(X_test)
        probs=model.predict_proba(X_test)
        metrics.conf_matrix(y_test,y_pred_test)
        metrics.roc_pr_curve(y_test,probs[:,1])
    
    elif "SVM" in model_name:
        from sklearn.linear_model import SGDClassifier as SGD

        print("Model: SVM")
        print("Train data")
        y_pred_train = model.predict(X_train)
        # probs=model.predict_proba(X_train)
        metrics.conf_matrix(y_train,y_pred_train)
        # metrics.roc_pr_curve(y_train,probs[:,1])
        print("Test data")
        y_pred_test = model.predict(X_test)
        # probs=model.predict_proba(X_test)
        metrics.conf_matrix(y_test,y_pred_test)
        # metrics.roc_pr_curve(y_test,probs[:,1])
    
    del X_train, X_test, y_train, y_test
    



