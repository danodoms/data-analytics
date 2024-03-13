# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:11:33 2024

@author: admin
"""

#import lib
import pandas as pd
import numpy as np




#load dataset
dataset = pd.read_csv('activity1.csv')
print(dataset.describe())



#create dependent & independent vars
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)




#handle missing data

#count number of missing values in each column
print(dataset.isnull().sum())
 
#drop missing values
dataset.dropna(inplace=True)

#replace missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])




#Reshape y to make it 2D
y = y.reshape(-1, 1)

#Correcting missing values for y
y_imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
y = y_imputer.fit_transform(y)

#Reshape y back to its original shape if necessary
y = y.flatten() #Flatten to 1D array if needed



# Data Encoding : Handle\Encode categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = "passthrough")
# x = np.array(ct.fit_transform(x))
# print(x)


# citiesColumn = x[:, 0]


#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

x[:, 0] = le.fit_transform(x[:, 0]) + 1



#Split the dataset for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)



#Feature Scaling - Standardization & Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[:, 0:] = scaler.fit_transform(x_train[:, 0:])
x_test[:, 0:] = scaler.fit_transform(x_test[:, 0:])

# y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.fit_transform(y_test.reshape(-1, 1)).flatten()

print(x_train)
print(x_test)



#Find and Remove Outliers
import matplotlib.pyplot as plt
plt.hist(dataset['Age'], bins=15)
plt.show()

#Quantile
lowerLimit = dataset['Age'].quantile(0.05)
lowerLimit
dataset[dataset['Age'] < lowerLimit]

upperLimit = dataset['Age'].quantile(0.95)
upperLimit
dataset[dataset['Age'] > upperLimit]



#def legendEncoder(old_value)
