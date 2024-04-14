# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:47:54 2024

@author: User
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
# df = pd.read_csv('StudentsPerformance.csv')
df = pd.read_csv('ytSample.csv')

# Assuming we want to perform ANOVA on the 'math score' column based on the 'race/ethnicity' column
# factor = 'race/ethnicity'
# response = 'math score'

factor = 'Company'
response = 'Salary'

# Get unique levels of the factor
levels = df[factor].unique()

# Calculate the mean of each level
means = {level: df[df[factor] == level][response].mean() for level in levels}

# Calculate the overall mean
overall_mean = df[response].mean()

# Calculate the between-group sum of squares (SSB)
SSB = sum([(means[level] - overall_mean)**2 * len(df[df[factor] == level]) for level in levels])

# Calculate the total sum of squares (SST)
SST = sum((df[response] - overall_mean)**2)

# Calculate the degrees of freedom for between groups (DFB) and within groups (DFW)
DFB = len(levels) - 1
DFW = len(df) - len(levels)

# Calculate the mean squares for between groups (MSB) and within groups (MSW)
MSB = SSB / DFB
MSW = SST / DFW

# Calculate the F-value
F_value = MSB / MSW

# Calculate the p-value
p_value = 1 - stats.f.cdf(F_value, DFB, DFW)

# Create a dataframe to store the ANOVA results
anova_results = pd.DataFrame({
    'Source of Variation': [f'Between {factor}'],
    'Sum of Squares': [SSB],
    'Degrees of Freedom': [DFB],
    'Mean Squares': [MSB],
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