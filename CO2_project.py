# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:16:54 2020

@author: KIIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
dataset=pd.read_csv('FuelConsumptionCo2.csv')
dataset.isnull().sum()

#selecting required columns
dataset = dataset.drop(['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB_MPG'],axis=1)

#dealing with categorical data an dummy variable trap
dummy=pd.get_dummies(dataset.FUELTYPE)
merged=pd.concat([dummy,dataset],axis=1)
final=merged.drop(['FUELTYPE','D'],axis=1)

#applying heatmap
figure = plt.figure(figsize=(5,5))
sns.heatmap(data=final.corr(),cmap='coolwarm',vmin=-1, vmax=1)

#creating indepedent and dependent variables
X = final.iloc[:,:-1]
y = final.iloc[:,6:7]

#feature scalling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)

#splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state =0)

#training model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting values
y_pred = regressor.predict(X_test)
print('Model Score : '+str(regressor.score(X_test,y_test)))
print('Mean Absolute Error:', np.mean(abs(y_pred - y_test)))
print('Mean Squared Error :', np.mean((y_pred - y_test)**2))
print('Root Mean Squared Error:',np.mean((y_pred - y_test)**2)**0.5)

import seaborn as sns
y_test=sc_y.inverse_transform(y_test)
y_pred=sc_y.inverse_transform(y_pred)
ax1=sns.distplot(y_test,hist=False,color='r',label='Test Set')
sns.distplot(y_pred,hist=False,color='b',label='Predicted Values',ax=ax1)

