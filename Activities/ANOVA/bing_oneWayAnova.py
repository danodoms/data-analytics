#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:17:44 2024

@author: danodoms
"""

import pandas as pd
import scipy.stats as stats

# Example DataFrame (replace with your actual data)
# data = {
#     'A': [10, 15, 20, 25],
#     'B': [12, 18, 22, 28],
#     'C': [8, 14, 18, 24],
#     'D': [11, 16, 21, 26]
# }

df = pd.read_csv("iris.csv")
target = "Species"
Y = df[target]
X = df.drop(target, axis=1)


# Perform ANOVA for each column
for col in X.columns:
    fvalue, pvalue = stats.f_oneway(X[col], Y)  # Replace 'Y' with your target column name
    print(f"Column '{col}': F-statistic = {fvalue:.4f}, p-value = {pvalue:.4f}")

# You can then analyze the p-values to decide which columns to keep or drop.
# Smaller p-values indicate stronger evidence against the null hypothesis (i.e., significant differences).
