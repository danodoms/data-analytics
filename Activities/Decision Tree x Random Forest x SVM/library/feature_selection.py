import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def cramers_v_analysis(data, target_column='Feedback', threshold=0.25, top_n=5, plot=True, figsize=None):
    """
    Perform Cramér's V analysis on a DataFrame.

    Parameters:
    data (DataFrame): Input DataFrame.
    target_column (str): Name of the target column.
    threshold (float): Threshold for high association.
    top_n (int): Number of top features to list.
    plot (bool): Whether to plot the heatmap.
    figsize (tuple): Size of the figure (width, height).

    Returns:
    dict: Dictionary containing the top features with high Cramér's V scores.
    """

    # Function to calculate Cramér's V
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        return chi2 / (n * (min(confusion_matrix.shape) - 1)) ** 0.5

    # Calculate Cramér's V for all pairs of features
    cramers_v_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != target_column and col2 != target_column:
                cramers_v_matrix.loc[col1, col2] = cramers_v(data[col1], data[col2])

    # Plot heatmap
    if plot:
        if figsize:
            plt.figure(figsize=figsize)
        sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Cramér's V Heatmap")
        plt.show()

    # Drop highly associated features
    to_drop = set()
    for col1 in cramers_v_matrix.columns:
        for col2 in cramers_v_matrix.columns:
            if col1 != col2 and cramers_v_matrix.loc[col1, col2] > threshold:
                to_drop.add(col2)

    # Extract top features with high Cramér's V scores
    top_features = cramers_v_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(top_n).to_dict()

    # Remove target column and dropped columns
    remaining_features = data.drop(columns=to_drop).drop(columns=target_column)

    return top_features, remaining_features

# Example usage:
# top_features, remaining_features = cramers_v_analysis(df_encoded, figsize=(12, 10))
# print("Top features with high Cramér's V scores:", top_features)
# print("Remaining features:", remaining_features.columns)
