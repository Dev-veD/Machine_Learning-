# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np


#importing Data
#reading from Data.csv file 
dataset = pd.read_csv("Data.csv")

#trimming out our required portion from dataset
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


#filling in missing values using Imputer class
from sklearn.preprocessing import Imputer

#creating an imputer object
imp = Imputer( missing_values = 'NaN', strategy = 'mean', axis=0 )

#fitting imputer object to our data
imp = imp.fit(X[:,1:3])

#fillng in missing values using transform function
X[:,1:3] = imp.transform(X[:,1:3])

#encoding variables 
from sklearn.preprocessing import OneHotEncoder , LabelEncoder 
#Encoding part done here
label_X = LabelEncoder()
label_Y = LabelEncoder()
X[:,0] = label_X.fit_transform(X[:,0])

onehot = OneHotEncoder( categorical_features = [0] )

X = onehot.fit_transform(X).toarray()

Y = label_Y.fit_transform(Y)
print(Y)

        #data spliting to training and testing

from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler

Sc = StandardScaler()
X_train = Sc.fit_transform(X_train)
X_test = Sc.transform(X_test)

plt.scatter(X_train[:,4],Y_train)

plt.show































