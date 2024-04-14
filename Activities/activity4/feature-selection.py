# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.preprocessing import LabelEncoder

def encode_dataframe(df):
    encoded_df = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    
    # Iterate through each column in the DataFrame
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    
    return encoded_df


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr





# filename = "climate_change.csv"
# target = "Temp"

filename = "Location1.csv"
target = "Power"

# filename = "laptop_data_cleaned.csv"
# target = "Price"


# Load the dataset
data = encode_dataframe(pd.read_csv(filename))

print(data.describe())

# Separate features (X) anspearmand target variable (y)
X = data.drop(target, axis=1)
y = data[target]

# Separate dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)





# 6. Perform Feature selection using Pearson Correlation Coefficient and Spearman's Correlation Coefficient for Predictors variables and Target Variables
# 7. Determine what are the significant and insignificant predictors (x) variables

# PEARSON AND SPEARMAN SCORES

# Calculate Spearman correlation coefficients
correlation_coefficients_spearman = data.corr(method='spearman')[target]

# Display Spearman correlation coefficients for each targetor
print("Spearman Correlation Coefficients:")
print(correlation_coefficients_spearman[:-1])  # Exclude the last entry (correlation with itself)

# Calculate Pearson correlation coefficients
correlation_coefficients_pearson = data.corr()[target]

# Display Pearson correlation coefficients for each targetor
print("Pearson Correlation Coefficients:")
print(correlation_coefficients_pearson[:-1])  # Exclude the last entry (correlation with itself)




# 8. Perform Pearson and Spearman’s Correlation Coefficient on the predictors (x)
# 9. Determine the highly correlated predictors and determine the predictors that must be removed from the dataset depending on a given threshold.

# Calculate Pearson correlation matrix for features in the training set
cor = X_train.corr()

# Plotting the heatmap for correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.title("Pearson Correlation Heatmap")
plt.show()


corr_features = correlation(X_train, 0.7)
len(set(corr_features))



# Calculate Spearman correlation matrix for features in the training set
cor = X_train.corr(method='spearman')

# Plotting the heatmap for correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.title("Spearman Correlation Heatmap")
plt.show()

corr_features = correlation(X_train, 0.7)
len(set(corr_features))


# Export the cleaned and featured-selected dataset on a .csv file.