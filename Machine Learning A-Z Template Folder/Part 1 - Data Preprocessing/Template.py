#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 01:25:47 2019

@author: divyanshu
"""


import pandas as pd
import numpy as np


#importing Data
#reading from Data.csv file 
dataset = pd.read_csv("Data.csv")

#trimming out our required portion from dataset
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#data spliting to training and testing

from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler

Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)

