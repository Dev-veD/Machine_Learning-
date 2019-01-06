#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 00:05:09 2019

@author: divyanshu
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelX = LabelEncoder()
X[:,3] = labelX.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features = [3])
X = onehot.fit_transform(X).toarray()
X = X[:,1:]
from sklearn.model_selection import train_test_split
X_train ,X_test,Y_train ,Y_test = train_test_split(X,Y,test_size=1/5,random_state =42)

#fitting dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)
print(Y_test-Y_pred)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) , values = X ,axis=1)
