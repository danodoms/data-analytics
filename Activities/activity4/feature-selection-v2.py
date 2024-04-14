# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:45:35 2024

@author: User
"""

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.preprocessing import LabelEncoder

def encode_dataframe(df):
    encoded_df = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    
    # Iterate through each column in the DataFrame
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    
    return encoded_df

# identify collinearity between predictors (x) for pearson dataframe (accepts negative or positive threshold)
def correlation_pearson(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    # corr_matrix = dataset.corr(method=method)
    
    for i in range(len(dataset.columns)):
        for j in range(i):
            if abs(dataset.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = dataset.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# identify collinearity between predictors (x) for spearman dataframe (accepts only positive threshold)
def correlation_spearman(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    # corr_matrix = dataset.corr(method=method)
    
    for i in range(len(dataset.columns)):
        for j in range(i):
            if (dataset.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = dataset.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr





# filename = "climate_change.csv"
# target = "Temp"

filename = "Location1.csv"
target = "Power/hour"

# filename = "Location1.csv feature_scaled.csv"
# target = "Power/hour"

# filename = "laptop_data_cleaned.csv"
# target = "Price"

# filename = "boston.csv"
# target = "MEDV"


# Load the dataset
data = encode_dataframe(pd.read_csv(filename))

print(data.describe())

# Separate features (X) anspearmand target variable (y)
X = data.drop(target, axis=1)
y = data[target]

# Separate dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)



# 6. Perform Feature selection using Pearson Correlation Coefficient and Spearman's Correlation Coefficient for Predictors variables and Target Variables
# 7. Determine what are the significant and insignificant predictorsD (x) variables

# PEARSON AND SPEARMAN SCORES
# Calculate Spearman correlation coefficients
correlation_coefficients_spearman = data.corr(method='spearman')[target]

# Display Spearman correlation coefficients for each target
print("Spearman Correlation Coefficients:")
print(correlation_coefficients_spearman[:-1])  # Exclude the last entry (correlation with itself)
spearmanInsignificant = correlation_coefficients_spearman[correlation_coefficients_spearman.abs() < 0.4].index.tolist()
spearmanSignificant = correlation_coefficients_spearman[(correlation_coefficients_spearman.abs() >= 0.4) & (correlation_coefficients_spearman.index != "Power/hour")].index.tolist()
correlation_coefficients_spearman = correlation_coefficients_spearman.drop(target)



# Calculate Pearson correlation coefficients

correlation_coefficients_pearson = data.corr()[target]

# Display Pearson correlation coefficients for each targetor
print("Pearson Correlation Coefficients:")
print(correlation_coefficients_pearson[:-1])  # Exclude the last entry (correlation with itself)
pearsonInsignificant = correlation_coefficients_pearson[correlation_coefficients_pearson.abs() < 0.4].index.tolist()
pearsonSignificant = correlation_coefficients_pearson[(correlation_coefficients_pearson.abs() >= 0.4) & (correlation_coefficients_pearson.index != "Power/hour")].index.tolist()
correlation_coefficients_pearson = correlation_coefficients_pearson.drop(target)


# data.drop(pearsonInsignificant, axis=1, inplace=True)

# 8. Perform Pearson and Spearman’s Correlation Coefficient on the predictors (x)

# Calculate Pearson correlation matrix for features in the training set
cor_pearson_heatmap = data.corr()

# Plotting the heatmap for correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cor_pearson_heatmap, annot=True, cmap=plt.cm.CMRmap_r)
plt.title(filename + " Pearson Correlation Heatmap")
plt.show()


corr_features_pearson = correlation_pearson(cor_pearson_heatmap, 0.8)
len(set(corr_features_pearson))




# Calculate Spearman correlation matrix for features in the training set
cor_spearman_heatmap = data.corr(method='spearman')

# Plotting the heatmap for correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cor_spearman_heatmap, annot=True, cmap=plt.cm.CMRmap_r)
plt.title(filename + " Spearman Correlation Heatmap")
plt.show()

corr_features_spearman = correlation_spearman(cor_spearman_heatmap, 0.8)
len(set(corr_features_spearman))

# 9. Determine the highly correlated predictors and determine the predictors that must be removed from the dataset depending on a given threshold.

# wind_power_generation.csv
features_to_remove = []
features_to_remove = ['winddirection_10m', 'temperature_2m', 'windgusts_10m', 'windspeed_10m', ]
data.drop(features_to_remove, axis=1, inplace=True)

# 10. Export the cleaned and featured-selected dataset on a .csv file.
data.to_csv(filename + ' feature_scaled.csv', index=False)