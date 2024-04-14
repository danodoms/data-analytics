#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 06:27:07 2024

@author: danodoms
"""

import pandas as pd
from sklearn.feature_selection import f_classif
from statsmodels.formula.api import ols  # for ANOVA

def anova_feature_selection(data, target_column):
  """
  Performs ANOVA feature selection on categorical data.

  Args:
      data: A pandas DataFrame containing the data.
      target_column: The name of the column containing the target variable.

  Returns:
      A pandas DataFrame with columns: 'feature', 'f_value', 'p_value'.
  """

  X = data.drop(target_column, axis=1)  # Features (categorical)
  y = data[target_column]  # Target variable

  # Create a list to store feature names, F-values, and p-values
  results = []

  for feature in X.columns:
    # One-hot encode the categorical feature (optional)
    # You might need to explore other encoding techniques for categorical data
    X_encoded = pd.get_dummies(X[[feature]], drop_first=True)

    # Use statsmodels formula API for ANOVA
    model = ols(f"{target_column} ~ {feature}", data=pd.concat([X_encoded, y], axis=1))
    f_value, p_value = f_classif(X_encoded, y)  # Alternative approach (less informative)

    results.append({"feature": feature, "f_value": f_value[0], "p_value": p_value})

  # Create a DataFrame with the results
  return pd.DataFrame(results)


df = pd.read_csv("StudentsPerformance.csv")

# Assuming your data is in a DataFrame named 'data'
target_column = "math_score"  # Replace with your actual target column name
scores_df = anova_feature_selection(df, target_column)

print(scores_df)
