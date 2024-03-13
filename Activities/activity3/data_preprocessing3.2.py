# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:51:15 2024

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:52:43 2024

@author: Henreh with ChatGPT
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
dataset = pd.read_csv('rice-yield-act3.csv')

# Separate the dataset into train data and test data for Rainfall_mm
train_data_rainfall = dataset.dropna(subset=['Rainfall_mm'])
test_data_rainfall = dataset[dataset['Rainfall_mm'].isnull()]

# Function to replace missing values of Rainfall_mm using linear regression
def replace_missing_rainfall(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Average_Temperature_C', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_train = train_data['Rainfall_mm']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Average_Temperature_C', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Rainfall_mm'] = y_pred
    return test_data

# Replace missing values of Rainfall_mm
test_data_rainfall = replace_missing_rainfall(train_data_rainfall, test_data_rainfall)

# Merge the modified test data for Rainfall_mm back into the original dataset using the index
dataset_imputed_rainfall = dataset.copy()
dataset_imputed_rainfall.loc[test_data_rainfall.index, 'Rainfall_mm'] = test_data_rainfall['Rainfall_mm']

# Print the dataset after replacing missing values of Rainfall_mm
print("Dataset after replacing missing values of Rainfall_mm:")
print(dataset_imputed_rainfall)

# Separate the dataset into train data and test data for Soil_pH
train_data_soil_ph = dataset_imputed_rainfall.dropna(subset=['Soil_pH'])
test_data_soil_ph = dataset_imputed_rainfall[dataset_imputed_rainfall['Soil_pH'].isnull()]

# Function to replace missing values of Soil_pH using linear regression
def replace_missing_soil_ph(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_train = train_data['Soil_pH']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Soil_pH'] = y_pred
    return test_data

# Replace missing values of Soil_pH
test_data_soil_ph = replace_missing_soil_ph(train_data_soil_ph, test_data_soil_ph)

# Merge the modified test data for Soil_pH back into the original dataset using the index
dataset_imputed_soil_ph = dataset_imputed_rainfall.copy()
dataset_imputed_soil_ph.loc[test_data_soil_ph.index, 'Soil_pH'] = test_data_soil_ph['Soil_pH']

# Print the dataset after replacing missing values of Soil_pH
print("\nDataset after replacing missing values of Soil_pH:")
print(dataset_imputed_soil_ph)

# Separate the dataset into train data and test data for Nitrogen_Content_kg_ha
train_data_nitrogen = dataset_imputed_soil_ph.dropna(subset=['Nitrogen_Content_kg_ha'])
test_data_nitrogen = dataset_imputed_soil_ph[dataset_imputed_soil_ph['Nitrogen_Content_kg_ha'].isnull()]

# Function to replace missing values of Nitrogen_Content_kg_ha using linear regression
def replace_missing_nitrogen(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_train = train_data['Nitrogen_Content_kg_ha']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Nitrogen_Content_kg_ha'] = y_pred
    return test_data

# Replace missing values of Nitrogen_Content_kg_ha
test_data_nitrogen = replace_missing_nitrogen(train_data_nitrogen, test_data_nitrogen)

# Merge the modified test data for Nitrogen_Content_kg_ha back into the original dataset using the index
dataset_imputed_nitrogen = dataset_imputed_soil_ph.copy()
dataset_imputed_nitrogen.loc[test_data_nitrogen.index, 'Nitrogen_Content_kg_ha'] = test_data_nitrogen['Nitrogen_Content_kg_ha']

# Print the dataset after replacing missing values of Nitrogen_Content_kg_ha
print("\nDataset after replacing missing values of Nitrogen_Content_kg_ha:")
print(dataset_imputed_nitrogen)

# Separate the dataset into train data and test data for Irrigation_Frequency_per_week
train_data_irrigation = dataset_imputed_nitrogen.dropna(subset=['Irrigation_Frequency_per_week'])
test_data_irrigation = dataset_imputed_nitrogen[dataset_imputed_nitrogen['Irrigation_Frequency_per_week'].isnull()]

# Function to replace missing values of Irrigation_Frequency_per_week using linear regression
def replace_missing_irrigation(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_train = train_data['Irrigation_Frequency_per_week']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Irrigation_Frequency_per_week'] = y_pred
    return test_data

# Replace missing values of Irrigation_Frequency_per_week
test_data_irrigation = replace_missing_irrigation(train_data_irrigation, test_data_irrigation)

# Merge the modified test data for Irrigation_Frequency_per_week back into the original dataset using the index
dataset_imputed_irrigation = dataset_imputed_nitrogen.copy()
dataset_imputed_irrigation.loc[test_data_irrigation.index, 'Irrigation_Frequency_per_week'] = test_data_irrigation['Irrigation_Frequency_per_week']

# Print the dataset after replacing missing values of Irrigation_Frequency_per_week
print("\nDataset after replacing missing values of Irrigation_Frequency_per_week:")
print(dataset_imputed_irrigation)

dataset_imputed_irrigation.to_csv('act3-dano-linear-regression.csv', index=False)

df = dataset_imputed_irrigation
df.describe()




# Direct Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_direct_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_direct_normalized.to_csv('directly-normalized-act3-dano-linear-regression.csv', index=False)




# # Standardization
# scalerSD = StandardScaler()
# df_standardized = scalerSD.fit_transform(df)
# df_standardized.to_csv('standardized-act3-dano-linear-regression.csv', index=False)

# # Normalization
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df_normalized = scaler.fit_transform(df)

# Standardization
scalerSD = StandardScaler()
df_standardized = pd.DataFrame(scalerSD.fit_transform(df), columns=df.columns)
df_standardized.to_csv('standardized-act3-dano-linear-regression.csv', index=False)


# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_standardized), columns=df_standardized.columns)
df_normalized.to_csv('normalized-act3-dano-linear-regression.csv', index=False)



