# -*- coding: utf-8 -*-
"""
Created on Thu Apr 9 11:21:46 2024

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define the feature names based on their indices
# feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
feature_names = ['Location', 'Company']

# Load the CSV file (assuming "iris.csv" contains the Iris dataset)
# data = pd.read_csv("iris.csv")
data = pd.read_csv("ytSample.csv")

# Extract the target variable and features // ASSUMES THAT Y or TARGET COLUMN IS IN LAST INDEX
y = data.iloc[:, -1].values
X = data.iloc[:, :-1].values

# Check if y is categorical and convert using label encoding
if isinstance(y[0], str):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
# Perform ANOVA test for each feature
p_values = []
for i in range(X.shape[1]):
    feature = X[:, i]
    mean_y = np.mean(y)
    mean_feature = np.mean(feature)
    ss_total = np.sum((y - mean_y) ** 2)
    ss_residual = np.sum((y - (mean_y + (feature - mean_feature) * np.mean(y / feature))) ** 2)
    ss_explained = ss_total - ss_residual
    f_value = ss_explained / ss_residual
    p_value = 1 - np.sum(np.array([1, 2, 3, 4]) * 2 * f_value / (X.shape[0] - 1)) * (-(X.shape[0] - 1) / 2)
    p_values.append(p_value)

# Select the features with p-value below a threshold (e.g., 0.05)
selected_features_indices = [i for i, p_value in enumerate(p_values) if p_value < 0.05]

# Map the selected feature indices to their names
selected_features = [feature_names[i] for i in selected_features_indices]

# Print ANOVA results for each feature
for i, p_value in enumerate(p_values):
    feature_name = f"Feature {i + 1}"
    print(f"{feature_name}: p-value = {p_value}")

# Print the selected features
print("Selected features:", selected_features_indices)
print("Selected features:", selected_features)

# Plot the p-values
plt.figure()
plt.bar(range(X.shape[1]), p_values)
plt.xlabel("Feature index")
plt.ylabel("p-value")
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title("ANOVA p-values for each feature")
plt.show()
