# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:10:19 2024

@author: admin
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
dataset = pd.read_csv('rice-yield-act3.csv')
dataset.shape # Get the structure, number of rows and columns
dataset.info() # Get the information/summary
dataset.isnull().sum() # Get the sum of all null



# Removing columns with nan values
columns_to_remove = ['Soil_pH', 'Nitrogen_Content_kg_ha', 'Irrigation_Frequency_per_week']
dataset_rainfall = dataset.drop(columns_to_remove, axis=1)



# Creating train and test dataframes
test_data = dataset_rainfall[dataset['Rainfall_mm'].isnull()]
train_data = dataset_rainfall.dropna() 
train_data.isnull().sum()



# Creating x and y training data
x_train_rainfall = train_data.drop("Rainfall_mm", axis=1)
y_train_rainfall = train_data['Rainfall_mm']



# Creating the Linear Regression Model
lr = LinearRegression()
lr.fit(x_train_rainfall, y_train_rainfall)



# Creating x test data
x_test = test_data.drop("Rainfall_mm", axis=1)
y_pred = lr.predict(x_test)



# Merging the y_pred data
test_data.loc[test_data.Rainfall_mm.isnull(), 'Rainfall_mm'] = y_pred
dataset_new = pd.concat([train_data, test_data], ignore_index=False)
dataset_new.sort_index(inplace=True)



# Standardization
from sklearn.preprocessing import StandardScaler
scalerSD = StandardScaler()
df_standardized = pd.DataFrame(scalerSD.fit_transform(dataset_new), columns=dataset_new.columns)
df_standardized.to_csv('1.csv', index=False)