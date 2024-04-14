import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("iris.csv")

# Extract the target variable and features
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Calculate Kendall's Tau correlation coefficient and p-value manually
correlations = []
for i in range(X.shape[1]):
    feature = X.iloc[:, i]
    n = len(feature)
    concordant_pairs = 0
    discordant_pairs = 0
    for j in range(n):
        for k in range(j + 1, n):
            if (y[j] > y[k] and feature[j] > feature[k]) or (y[j] < y[k] and feature[j] < feature[k]):
                concordant_pairs += 1
            elif (y[j] > y[k] and feature[j] < feature[k]) or (y[j] < y[k] and feature[j] > feature[k]):
                discordant_pairs += 1
    tau = (concordant_pairs - discordant_pairs) / (0.5 * n * (n - 1))
    # Calculate p-value using permutation test (optional, you can skip this part if not needed)
    all_pairs = concordant_pairs + discordant_pairs
    permuted_taus = []
    for _ in range(10):  # Number of permutations
        permuted_y = np.random.permutation(y)
        permuted_concordant_pairs = sum(
            (permuted_y[j] > permuted_y[k] and feature[j] > feature[k]) or (permuted_y[j] < permuted_y[k] and feature[j] < feature[k])
            for j in range(n)
            for k in range(j + 1, n)
        )
        permuted_tau = (permuted_concordant_pairs - (all_pairs - permuted_concordant_pairs)) / (0.5 * n * (n - 1))
        permuted_taus.append(permuted_tau)
    p_value = np.mean(np.abs(permuted_taus) >= np.abs(tau))
    correlations.append((i, tau, p_value))

# Sort the features by correlation coefficient and select the top k features
k = 5  # Select the top 5 features
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
selected_features_indices = [x[0] for x in correlations[:k]]

# Map the selected feature indices to their names
feature_names = list(X.columns)
selected_features = [feature_names[i] for i in selected_features_indices]

# Print Kendall's Tau and p-value for each feature
print("Selected features:")
for i, (index, tau, p_value) in enumerate(correlations):
    feature_name = f"Feature {index + 1}"
    if index in selected_features_indices:
        print(f"{feature_name}: Tau = {tau}, p-value = {p_value}")

# Print the selected features with indices and names
print("Selected features indices:", selected_features_indices)
print("Selected features:", selected_features)

# Plot the p-values
p_values = [x[2] for x in correlations]
plt.figure()
plt.bar(range(len(p_values)), p_values)
plt.xlabel("Feature index")
plt.ylabel("p-value")
plt.title("Kendall's Tau p-values for each feature")
plt.show()
