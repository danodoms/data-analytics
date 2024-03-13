# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:39:37 2024

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:44:13 2024

@author: User
"""

#Activity 3

#1.) Converting categorical to nominal in x column

#2.) convert column four to numeric

#3.) Separate the null values

#4.) Drop the Null values from the dataframe and consider as train data

#5.) check if there is an existing null in the train data

#6.) Create X train and Y train from the train data

#7.) Build the Model

#8.) create the x_test from the test_data

#9.) Apply the model on x_test and predicting missing values for age

#10.) Replace the Missing values with predicted values

#11.) Merge the modified test data (with imputed Age values) back into the original dataset

#12.) Sort the dataset by index to ensure it's in the original order

# Importing libraries

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv('dataset2.csv')

# Get the structure, number of rows and columns
dataset.shape

# Get the information/summary
dataset.info()

# Get the sum of all null
dataset.isnull().sum()

# Print dataset
dataset

# 1.) Converting categorical to nominal in x column 8

# Label encoding //convert categorical to nominal | 0,1,2,3.. in column @ of x from sklearn.preprocessing import LabelEncoder

# Apply LabelEncoder to column 8
le = LabelEncoder()
dataset.iloc[:, 0] = le.fit_transform(dataset.iloc[:, 0])

# 2.) Convert column four to numeric
le = LabelEncoder()
dataset.iloc[:, 3] = le.fit_transform(dataset.iloc[:, 3])

# 3.) Separate the null values
# Separate the NULL values from 'Age' feature
test_data = dataset[dataset['Age'].isnull()]

# 4.) Drop the Null values from the dataframe and consider as train data
dataset.dropna(inplace=True)

# 5.) Check if there is an existing null in the train data
dataset.isnull().sum()

# Check the structure of the train data
dataset.shape

# 6.) Create X_train and Y_train from the train data
# For Age
# Y_train means rows from the dataset['Age'] with non-null values
y_train_age = dataset['Age']

# X_train means dataset except dataset['Age'] features with non-null values
x_train_age = dataset.drop("Age", axis=1)

# 7.) Build the Model
lr = LinearRegression()
# Train the model on the train dataset
lr.fit(x_train_age, y_train_age)

# 8.) Create the X_test from the test_data
# For age
# X_test means dataset except dataset['Age'] feature with NULL values
x_test_age = test_data.drop("Age", axis=1)

# 9.) Apply the model on X_test and predict missing values for age
y_pred_age = lr.predict(x_test_age)

# 10.) Replace the Missing values with predicted values
test_data.loc[test_data.Age.isnull(), 'Age'] = y_pred_age

# 11.) Merge the modified test_data (with imputed Age values) back into the original dataset
dataset_new = pd.concat([dataset, test_data], ignore_index=True)

# 12.) Sort the dataset by index to ensure it's in the original order
dataset_new.sort_index(inplace=True)