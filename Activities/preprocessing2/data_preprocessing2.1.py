# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:04:06 2024

@author: Dominador Dano Jr. Gikan kang sir Neil Arellano Mutia
"""
#Activity 2
#1.) creating independent x and dependent y variables
#2.) Filling out missing data using mean x
#3.) filling our missing data using most_frequent in y
#4.) converting x and y into one Dataset/dataframe
#5.) Determine lower and upper limit of age using quantile
#6.) Remove the outliers from the dataset AGE
#7.) Determine lower and upper limit of SALARY using quantile
#8.) Remove the outliers from the dataset SALARY
#9.) creating independent x and dependent y variables
#10.) Converting categorical to nominal in x column 0
#11.) Converting categorical to nominal in y
#12.) feature scaling - Standardization & Normalization for x
#13.) feature scaling - Standardization & Normalization for y
#14.) Convert x and y into one DataFrame
#15.) Export dataset_cleaned to CSV

# Importing libraries----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset-----------------------------------------------------------------
dataset = pd.read_csv('dataset2.csv')
print(dataset.describe())



# 1. Creating independent x and dependent y variables------------------------------ 1.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)



# 2. Filling missing data using mean----------------------------------------------- 2.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])



# 3. Filling missing data using most frequent in y--------------------------------- 3.
imputer_y = SimpleImputer(strategy='most_frequent')
y = imputer_y.fit_transform(y.reshape(-1, 1)).ravel()

# Convert x and y into one DataFrame----------------------------------------------- 4.
merged_data = np.hstack((x, y.reshape(-1, 1)))
columns = list(dataset.columns[:-1]) + ['Eligible_for_bonus']
merged_dataset = pd.DataFrame(merged_data, columns=columns)

# Identify and remove outliers for 'Age'------------------------------------------- 5. & 6.
age_lower_limit = merged_dataset['Age'].quantile(0.05)
age_upper_limit = merged_dataset['Age'].quantile(0.95)
dataset_cleaned_age = merged_dataset.loc[(merged_dataset['Age'] >= age_lower_limit) & (merged_dataset['Age'] <= age_upper_limit)]

# Identify and remove outliers for 'Salary'---------------------------------------- 7. & 8.
salary_lower_limit = dataset_cleaned_age['Salary'].quantile(0.05)
salary_upper_limit = dataset_cleaned_age['Salary'].quantile(0.95)
dataset_cleaned = dataset_cleaned_age.loc[(dataset_cleaned_age['Salary'] >= salary_lower_limit) & (dataset_cleaned_age['Salary'] <= salary_upper_limit)]




# viewing the graphs of age and salary data
plt.hist(dataset['Age'],bins=15) #bins = number of bar graph / intervals
plt.show()

plt.hist(dataset['Salary'],bins=15) #bins = number of bar graph / intervals
plt.show()

# 4. Creating independent x and dependent y variables after outlier removal-------- 9.
new_x = dataset_cleaned.iloc[:, :-1].values
new_y = dataset_cleaned.iloc[:, -1].values
print(new_x)
print(new_y)



# 5. Converting categorical to nominal in x column 0------------------------------- 10.
le = LabelEncoder()
new_x[:, 0] = le.fit_transform(new_x[:, 0])

# 6. Converting categorical to nominal in y---------------------------------------- 11.
new_y = le.fit_transform(new_y)
print(new_y)



# 7. Feature scaling - Standardization for x--------------------------------------- 12.
scaler = StandardScaler()
new_x[:, :3] = scaler.fit_transform(new_x[:, :3])



# 8. Feature scaling - Normalization for y----------------------------------------- 13.
new_y = scaler.fit_transform(new_y.reshape(-1, 1)).flatten()





#quantile for Age--------------------------------------------------------------

# Lower percentiles (e.g., 1st or 5th percentile): These are useful for identifying outliers 
# at the lower end of the distribution. They are often employed when data is expected to have 
# a lower bound or when you want to identify extreme low values.
age_lowerLimit = dataset['Age'].quantile(0.05)
age_lowerLimit

# Upper percentiles (e.g., 95th or 99th percentile): These are useful for identifying outliers 
# at the upper end of the distribution. They are often used when you want to detect extreme 
# high values or when the data may have an upper limit
age_upperLimit = dataset['Age'].quantile(0.95)
age_upperLimit



# Scatter plot of age column---------------------------------------------------
plt.scatter(dataset.index, dataset['Age'], label='Age')

# Plot outliers
outliers = dataset[(dataset['Age'] > age_upperLimit ) | (dataset['Age'] < age_lowerLimit)]
filteredData = dataset[(dataset['Age'] < age_upperLimit ) & (dataset['Age'] > age_lowerLimit)]
plt.scatter(outliers.index, outliers['Age'], color='red', label='Outliers')

# Draw lines for upper and lower limits
for index, row in outliers.iterrows():
    plt.axhline(y=row['Age'], color='gray', linestyle='--')

plt.axhline(y=age_lowerLimit, color='green', linestyle='--', label='Lower Limit')
plt.axhline(y=age_upperLimit, color='yellow', linestyle='--', label='Upper Limit')
plt.xlabel('Index')
plt.ylabel('Age')
plt.title('Scatter Plot of Age with Outliers')
plt.legend()
plt.grid(True)
plt.show()



#quantile for Salary-----------------------------------------------------------

# Lower percentiles (e.g., 1st or 5th percentile): These are useful for identifying outliers 
# at the lower end of the distribution. They are often employed when data is expected to have 
# a lower bound or when you want to identify extreme low values.
salary_lowerLimit = dataset['Salary'].quantile(0.05)
salary_lowerLimit

# Upper percentiles (e.g., 95th or 99th percentile): These are useful for identifying outliers 
# at the upper end of the distribution. They are often used when you want to detect extreme 
# high values or when the data may have an upper limit
salary_upperLimit = dataset['Salary'].quantile(0.95)
salary_upperLimit







# Scatter plot of age column---------------------------------------------------
plt.scatter(dataset.index, dataset['Salary'], label='Salary')

# Plot outliers
outliers1 = dataset[(dataset['Salary'] > salary_upperLimit ) | (dataset['Salary'] < salary_lowerLimit)]
filteredData1 = dataset[(dataset['Salary'] < salary_upperLimit ) & (dataset['Salary'] > salary_lowerLimit)]
plt.scatter(outliers1.index, outliers1['Salary'], color='red', label='Outliers')

# Draw lines for upper and lower limits
for index, row in outliers1.iterrows():
    plt.axhline(y=row['Salary'], color='gray', linestyle='--')

plt.axhline(y=salary_lowerLimit, color='green', linestyle='--', label='Lower Limit')
plt.axhline(y=salary_upperLimit, color='yellow', linestyle='--', label='Upper Limit')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.title('Scatter Plot of Salary with Outliers')
plt.legend()
plt.grid(True)
plt.show()




# viewing the graphs of age and salary data after processing
plt.hist(filteredData['Age'],bins=15) #bins = number of bar graph / intervals
plt.show()

plt.hist(filteredData1['Salary'],bins=15) #bins = number of bar graph / intervals
plt.show()





# Convert x and y into one DataFrame----------------------------------------------- 14.
merged_data = np.hstack((new_x, new_y.reshape(-1, 1)))
columns = list(dataset_cleaned.columns[:-1]) + ['Target']
merged_new_dataset = pd.DataFrame(merged_data, columns=columns)

# Export dataset cleaned to CSV---------------------------------------------------- 15.
merged_new_dataset.to_csv('processed_dataset.csv', index=False)