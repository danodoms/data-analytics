#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:39:19 2024

@author: danodoms
"""
import pandas as pd
import numpy as np
from scipy import stats
from decimal import Decimal

# Load the dataset
df = pd.read_csv('ytSample.csv')

# Assuming we want to perform ANOVA on the 'math score' column based on the 'race/ethnicity' column
factor = 'Company'
response = 'Salary'

# Get unique levels of the factor
levels = df[factor].unique() 

# Calculate the mean of each level
means = {level: df[df[factor] == level][response].mean() for level in levels}


# Calculate the degrees of freedom for between groups (DFB) and within groups (DFW)
DFB = len(levels) - 1
DFW = len(df) - len(levels)


def get_within_group_variance():
    # Calculate the sum of squares within groups (SSW)
    SSW = sum((df[df[factor] == level][response] - means[level]).pow(2).sum() for level in levels)
   
    N = len(df[response]) # Total number of observations
    
    k = len(levels)  # Number of groups (unique levels)
    
    # Degrees of freedom for within-group variation
    df_within = N - k
    within_group_variance = SSW/df_within

    return within_group_variance

def get_between_group_variance():
    # Calculate the overall mean
    overall_mean = df[response].mean()
    
    # Calculate the between-group sum of squares (SSB)
    SSB = sum([(means[level] - overall_mean)**2 * len(df[df[factor] == level]) for level in levels])
    
    # # Calculate the mean squares for between groups (MSB) and within groups (MSW)
    MSB = SSB / DFB
    
    return MSB


BG_Variance = get_between_group_variance()
WG_Variance = get_within_group_variance()

# Calculate the F-value
F_value =BG_Variance / WG_Variance

# Calculate the p-value
p_value = 1 - (stats.f.cdf(F_value, DFB, DFW))

# Create a dataframe to store the ANOVA results
anova_results = pd.DataFrame({
    'Source of Variation': [f'Between {factor}'],
    # 'Sum of Squares': [SSB],
    # 'Degrees of Freedom': [DFB],
    # 'Mean Squares': [MSB],
    'F-Value': [F_value],
    'p-value': [p_value]
})

# Print the ANOVA results dataframe
print(anova_results)

# Interpret the results
if p_value < 0.05:
    print("The result is statistically significant. There are significant differences between the groups.")
else:
    print("The result is not statistically significant. There are no significant differences between the groups.")

