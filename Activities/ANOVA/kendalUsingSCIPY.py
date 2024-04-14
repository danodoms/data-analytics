# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:43:25 2024

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Load the dataset
df = pd.read_csv("iris.csv")

# Extract the target variable and features
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Calculate Kendall's Tau correlation coefficient and p-value for each feature
correlations = []
for i in range(X.shape[1]):
    feature = X.iloc[:, i]
    tau, p_value = kendalltau(y, feature)
    correlations.append((i, tau, p_value))

# Sort the features by correlation coefficient and select the top k features
k = 5  # Select the top 5 features
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
selected_features_indices = [x[0] for x in correlations[:k]]

# Map the selected feature indices to their names
feature_names = list(X.columns)
selected_features = [feature_names[i] for i in selected_features_indices]

# Print Kendall's Tau and p-value for each feature
print("Selected features:")
for i, (index, tau, p_value) in enumerate(correlations):
    feature_name = f"Feature {index + 1}"
    if index in selected_features_indices:
        print(f"{feature_name}: Tau = {tau}, p-value = {p_value}")

# Print the selected features with indices and names
print("Selected features indices:", selected_features_indices)
print("Selected features:", selected_features)

# Plot the p-values
p_values = [x[2] for x in correlations]
plt.figure()
plt.bar(range(len(p_values)), p_values)
plt.xlabel("Feature index")
plt.ylabel("p-value")
plt.title("Kendall's Tau p-values for each feature")
plt.show()
