#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 22:44:32 2019

@author: divyanshu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)
Y_pred = regressor.predict(X_test)

#visuallising
#plt.scatter(X_train ,Y_train ,color = 'red')
plt.scatter(X_test , Y_test ,color = 'red')
#plt.scatter(X_test , Y_pred , color = 'pink')
plt.plot(X_train , regressor.predict(X_train) , color = 'blue')
plt.xlabel("Years of experience")
plt.ylabel("Salaries")
plt.title("Regression ")
plt.show()