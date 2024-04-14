import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Define the feature names based on their indices
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# Load the CSV file (assuming "iris.csv" contains the Iris dataset)
data = pd.read_csv("iris.csv")

# Extract the target variable and features
y = data.iloc[:, -1].values
X = data.iloc[:, :-1].values

def convert_to_normal_numbers(array, decimal_places=5):
  return np.around([float(num) for num in array], decimals=decimal_places)


# Check if y is categorical and convert using label encoding
if isinstance(y[0], str):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Perform ANOVA test for each feature
p_values = []
for i in range(X.shape[1]):
    f_value, p_value = f_oneway(*[X[y == k, i] for k in np.unique(y)])
    p_values.append(p_value)

new_p_values = convert_to_normal_numbers(p_values)

# Select the features with p-value below a threshold (e.g., 0.05)
selected_features_indices = [i for i, p_value in enumerate(new_p_values) if p_value > 0.05]
# selected_features_indices = [i for i, p_value in enumerate(p_values)]

# Map the selected feature indices to their names
selected_features = []
if selected_features_indices:
    selected_features = [feature_names[i] for i in selected_features_indices]

# Print ANOVA results for each feature
for i, p_value in enumerate(new_p_values):
    feature_name = f"Feature {i + 1}"
    print(f"{feature_name}: p-value = {p_value}")

# Print the selected features
print("Selected features:", selected_features_indices)
print("Selected features:", selected_features)

# Plot the p-values
plt.figure()
plt.bar(range(X.shape[1]), p_values)
plt.xlabel("Feature index")
plt.ylabel("p-value")
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title("ANOVA p-values for each feature")
plt.show()