
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
dataset = pd.read_csv('activity1.csv')
print(dataset.describe())

#create dependent & independent variable vectors
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#handle missing data

#print the total number of missing values in each column
print(dataset.isnull().sum())

# replace missing values
from sklearn.impute import SimpleImputer

x_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_imputer.fit(x[:, 1:3])
x[:, 1:3] = x_imputer.transform(x[:, 1:3])

# Reshape y to make it 2D
y = y.reshape(-1, 1)

# Correcting missing values for y
y_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
y = y_imputer.fit_transform(y)

# Reshape y back to its original shape if necessary
y = y.flatten()  # Flatten to 1D array if needed

# Data Encoding : Handle\Encode categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder= "passthrough")
x = np.array(ct.fit_transform(x))
print(x)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Split the dataset for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Feature Scaling - Standardization & Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[:,4:] = scaler.fit_transform(x_train[:,4:])
x_test[:,4:] = scaler.fit_transform(x_test[:,4:])

# Fit and transform the training data for the target variable (y)
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

# Transform the test data for the target variable using the scaler fitted on the training data
y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

print(x_train)
print(x_test)
