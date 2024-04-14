import pandas as pd
import numpy as np
from scipy import stats

def calculate_anova(df, response):
    # Define the target variable (response) and get unique levels
    target = response
    levels = df.columns[df.columns != target]

    # Calculate ANOVA for each feature
    results = []
    for factor in levels:
        # Calculate the mean for each level of the factor
        means = {level: df[df[factor] == level][target].mean() for level in df[factor].unique()}
        
        # Calculate degrees of freedom
        DFB = len(df[factor].unique()) - 1
        DFW = len(df) - len(df[factor].unique())
        
        # Calculate sum of squares within groups (SSW)
        SSW = sum((df[df[factor] == level][target] - means[level]).pow(2).sum() for level in df[factor].unique())
        
        # Calculate sum of squares between groups (SSB)
        SSB = sum([(means[level] - df[target].mean())**2 * len(df[df[factor] == level]) for level in df[factor].unique()])
        
        # Calculate between-group variance and within-group variance
        BG_Variance = SSB / DFB
        WG_Variance = SSW / DFW
        
        # Calculate F-value
        F_value = BG_Variance / WG_Variance
        
        # Calculate p-value
        p_value = 1 - (stats.f.cdf(F_value, DFB, DFW))
        
        # Determine if the result is significant
        remark = "Significant" if p_value < 0.05 else "Not Significant"
        
        # Store results
        results.append({
            'Feature': factor,
            'F-Value': F_value,
            'p-value': "%f" % p_value,
            'Remark': remark
        })
    
    # Create a dataframe to store the results
    anova_results = pd.DataFrame(results)
    
    return anova_results

# Load the dataset
df = pd.read_csv('ytSample.csv')
# df = pd.read_csv('StudentsPerformance.csv')
# df = pd.read_csv('House_Rent_Dataset.csv')

# Specify the target variable
response = 'Salary'
# response = 'math score'
# response = 'Rent'

# Calculate ANOVA for each feature
anova_results = calculate_anova(df, response)

# Print the ANOVA results
print(anova_results)
