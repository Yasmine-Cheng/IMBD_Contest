# -*- coding: utf-8 -*-
from pandas import read_csv

from keras.layers import Lambda, Input, Dense, Reshape, RepeatVector, Dropout
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.constraints import unit_norm, max_norm

from scipy import stats
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('../../projectB/train1/train1.csv')
df.fillna(0)

training_feature = df.iloc[:, :-1].values
ground_truth_r = df.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(training_feature)
training_feature = scaler.transform(training_feature)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_feature, ground_truth_r, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor( random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mean_squared_error_regr = mean_squared_error(y_test, y_pred)
r2_score_regr = r2_score(y_test, y_pred)
print(mean_squared_error_regr, r2_score_regr)

from xgboost.sklearn import XGBRegressor
parameters = {
                'n_estimators': [100,200], # default 100
                'max_depth': [3, 6, 9],
                'gamma': [0, 0.1],
                'subsample': [0.8, 0.9, 1],
                'reg_alpha': [0, 0.1, 0.3, 1], # L1 regularization, default 0
                'reg_lambda': [0, 1], # L2 regularization, default 1
                'learning_rate': [0.1, 0.3],
                'nthread': [20],
                'seed': [1]
             }

model = XGBRegressor()
# grid_obj = GridSearchCV(model, parameters, cv=10, scoring='r2')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mean_squared_error_xg = mean_squared_error(y_test, y_pred)
r2_score_xg = r2_score(y_test, y_pred)
print(mean_squared_error_xg, r2_score_xg)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
mean_squared_error_reg = mean_squared_error(y_test, y_pred)
r2_score_reg = r2_score(y_test, y_pred)
print(mean_squared_error_reg, r2_score_reg)

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mean_squared_error_linear = mean_squared_error(y_test, y_pred)
r2_score_linear = r2_score(y_test, y_pred)
print(mean_squared_error_linear, r2_score_linear)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mean_squared_error_ridge = mean_squared_error(y_test, y_pred)
r2_score_ridge = r2_score(y_test, y_pred)
print(mean_squared_error_ridge, r2_score_ridge)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=10, random_state=1)
er = VotingRegressor([('lr', r1), ('rf', r2)])
er.fit(X_train, y_train)
y_pred = er.predict(X_test)
mean_squared_error_voting = mean_squared_error(y_test, y_pred)
r2_score_voting = r2_score(y_test, y_pred)
print(mean_squared_error_voting, r2_score_voting)
