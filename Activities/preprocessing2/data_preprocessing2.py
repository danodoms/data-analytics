#Activity 2
#1.) creating independent x and dependent y variables
#2.) Filling out missing data using mean x
#3.) filling our missing data using most_frequent in y
#4.) Converting categorical to nominal in x column 0
#5.) Converting categorical to nominal in y
#6.) splitting data into training and test
#7.) feature scaling - Standardization & Normalization for x
#8.) feature scaling - Standardization & Normalization for y


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
dataset = pd.read_csv('dataset2.csv')
print(dataset.describe())

#1.) creating independent x and dependent y variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)



#2.) Filling out missing data using mean x
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])



#3.) filling our missing data using most_frequent in y
imputer = SimpleImputer(strategy='most_frequent')
dataset.iloc[:, 3] = imputer.fit_transform(dataset.iloc[:, 3].values.reshape(-1, 1)).ravel()



#4.) Converting categorical to nominal in x column 0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0]) + 1



#5.) Converting categorical to nominal in y
y = le.fit_transform(y)



#6.) splitting data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 1)




#7.) feature scaling - Standardization & Normalization for x
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[:, 0:] = scaler.fit_transform(x_train[:, 0:])
x_test[:, 0:] = scaler.fit_transform(x_test[:, 0:])



#8.) feature scaling - Standardization & Normalization for y
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.fit_transform(y_test.reshape(-1, 1)).flatten()



# viewing the graph of age data
plt.hist(dataset['Age'],bins=15) #bins = number of bar graph / intervals
plt.show()



#quantile

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



salary_lowerLimit = dataset['Salary'].quantile(0.05)
salary_lowerLimit

salary_upperLimit = dataset['Salary'].quantile(0.95)
salary_upperLimit


# Scatter plot of age column
plt.scatter(dataset.index, dataset['Age'], label='Age')

# Plot outliers
outliers = dataset[(dataset['Age'] > age_upperLimit ) | (dataset['Age'] < age_lowerLimit)]
plt.scatter(outliers.index, outliers['Age'], color='red', label='Outliers')
# Draw lines for upper and lower limits
for index, row in outliers.iterrows():
    plt.axhline(y=row['Age'], color='gray', linestyle='--')

plt.axhline(y=age_lowerLimit, color='orange', linestyle='--', label='Lower Limit')
plt.axhline(y=age_upperLimit, color='yellow', linestyle='--', label='Upper Limit')
plt.xlabel('Index')
plt.ylabel('Age')
plt.title('Scatter Plot of Age with Outliers')
plt.legend()
plt.grid(True)
plt.show()



# Scatter plot of salary column 
plt.scatter(dataset.index, dataset['Salary'], label='Salary')

# Plot outliers
outliers = dataset[(dataset['Salary'] > salary_upperLimit ) | (dataset['Salary'] < salary_lowerLimit)]
plt.scatter(outliers.index, outliers['Salary'], color='red', label='Outliers')
# Draw lines for upper and lower limits
for index, row in outliers.iterrows():
    plt.axhline(y=row['Salary'], color='gray', linestyle='--')

plt.axhline(y=salary_lowerLimit, color='orange', linestyle='--', label='Lower Limit')
plt.axhline(y=salary_upperLimit, color='yellow', linestyle='--', label='Upper Limit')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.title('Scatter Plot of Salary with Outliers')
plt.legend()
plt.grid(True)
plt.show()




x_combined = np.concatenate((x_train, x_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)



#Export processed data to csv
x_df = pd.DataFrame(x_combined, columns=["City", "Age", "Salary"])
y_df = pd.DataFrame(y_combined, columns=["Eligible for Bonus"])

# Concatenate processed DataFrames along columns (axis=1)
combined_df = pd.concat([x_df, y_df], axis=1)

cleaned_1 = combined_df[(combined_df['Salary'] > salary_upperLimit ) | (combined_df['Salary'] < salary_lowerLimit)]


# Export the combined DataFrame to a CSV file
combined_df.to_csv('processed_data.csv', index=False)