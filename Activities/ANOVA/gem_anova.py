#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 05:47:06 2024

@author: danodoms
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Assuming your data is in a CSV file
data = pd.read_csv("StudentsPerformance.csv")

# Separate features (X) and target variable (y)
data.drop("reading score", axis=1)
data.drop("writing score", axis=1)
X = data.drop("math score", axis=1)  # Replace "target_column" with your actual target column name
y = data["math score"]

# Create a selector object using f_classif (ANOVA F-value)
selector = SelectKBest(f_classif, k=10)  # k is the number of features to select

# Fit the selector on your data
selector.fit(X, y)

# Get the transformed features (X_new) containing the selected features
X_new = selector.transform(X)

# Get feature scores and p-values
feature_scores = list(zip(selector.scores_, selector.pvalues_))


# ANALYZE RESULTS

# Print feature importance scores
print("Feature importance:", feature_scores)

# Visualize feature scores (optional)
import matplotlib.pyplot as plt

plt.bar(X.columns, [x[0] for x in feature_scores])
plt.xlabel("Features")
plt.ylabel("ANOVA F-value")
plt.show()

