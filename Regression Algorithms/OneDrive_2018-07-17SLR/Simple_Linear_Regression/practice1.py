# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 12:16:49 2018

@author: v-sojag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mat

dataset = pd.read_csv('Salary_Data.csv')
x= dataset.iloc[:, :-1].values
y=dataset.iloc[:,1].values
#print(x)
#print(y)
                 
# Split the data into train and test on 80:20 ratio     
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#SLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)



