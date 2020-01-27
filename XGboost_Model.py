# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:02:14 2019

@author: Sandy Sun
"""

import nltk
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# Feature
x_train=pd.read_pickle("all_features_train_nov5.pkl")
x_train.drop(columns=['id'],inplace=True)

y_train = pd.read_pickle("y_train_nov5.pkl").values

x_test_full = pd.read_pickle("all_features_test_nov5.pkl")
x_test = x_test_full.drop(columns=['id'])

feature_names = list(x_train.columns.values)
print("Features: {}".format(feature_names))


ft=pd.read_pickle("feature_importances_1109.pkl")

def extract_pruned_features(feature_importances, min_score=200):
    # feature_l=[]
    column_slice = feature_importances[feature_importances['fscore'] > min_score]
    return list(column_slice['feature'].values)
pruned_featurse = extract_pruned_features(ft, min_score=500)
x_train = x_train.loc[:,pruned_featurse]
x_train = x_train.values

x_test = x_test.loc[:,pruned_featurse]
x_test = x_test.values

# Paramater
RS = 12357
ROUNDS = 500
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['silent'] = 1
params['seed'] = RS
params['eta'] = 0.18
params['max_depth'] = 11

kf = KFold(5, shuffle=True, random_state=RS)
xgb_preds=[]

for train, test in kf.split(x_train):
    train_X, valid_X = x_train[train], x_train[test]
    train_y, valid_y = y_train[train], y_train[test]
    
    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(x_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(params, d_train, 500,  watchlist, maximize=True, verbose_eval=50) #early_stopping_rounds=50)
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
    
    
import time
submission=time.time()
preds = np.asarray(xgb_preds)
preds = np.mean(preds, axis=0)
sub=pd.DataFrame()
sub['id'] = x_test_full['id']
sub['same_source'] = preds
sub.to_csv( "xgb_seed{}_n{}_sb{}.csv".format(RS, ROUNDS,submission), index=False)





