import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# # Load CSV
# df = pd.read_csv('data/onlinefoods.csv')


# label_encoder = LabelEncoder()
# # Apply Label Encoding to the categorical columns
# for col in df.columns:
#     if df[col].dtype == 'object':
#         df[col] = label_encoder.fit_transform(df[col])

# # One-hot encode categorical variables
# df_encoded = pd.get_dummies(df, columns=['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
#                                           'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 
#                                           'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 
#                                           'COUGHING', 'SHORTNESS OF BREATH', 
#                                           'SWALLOWING DIFFICULTY', 'CHEST PAIN'],
#                               drop_first=True)

df_encoded = pd.read_csv("data/onlinefoods_encoded.csv")

isnull = df_encoded.isnull().sum()

# Split data into X (features) and Y (target variable)
X = df_encoded.drop('Feedback', axis=1)
Y = df_encoded['Feedback']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test_scaled)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(Y_test, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)

# Calculate correlation matrix
corr_matrix = df_encoded.corr()

# Create a heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Perform chi-square test for each feature
feature_significance = {}
for col in X_train.columns:
    contingency_table = pd.crosstab(X_train[col], Y_train)
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    feature_significance[col] = {"Significance": "Significant" if p_val < 0.05 else "Not Significant",
                                  "p-value": p_val}

# Print significance and p-values of each feature
print("\nFeature Significance and p-values:")
for feature, values in feature_significance.items():
    print(f"{feature}: {values['Significance']}, p-value: {values['p-value']}")


# Create DataFrame to store chi-square test results
chi2_df = pd.DataFrame(feature_significance).T
chi2_df.index.name = 'Feature'
chi2_df.reset_index(inplace=True)

# Display chi2_df in the variable explorer
chi2_df

# Drop not significant features
significant_features = [feat for feat, values in feature_significance.items() if values["Significance"] == "Significant"]
df_filtered = df_encoded[significant_features + ['Feedback']]

# Save filtered data to CSV
df_filtered.to_csv('filtered_onlinefoods.csv', index=False)
