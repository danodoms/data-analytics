import numpy as np
import pandas as pd

df = pd.read_csv("Earthquake.csv")
print(df.describe())



x = df.iloc[:, :-1]
y = df.iloc[:, -1] 



# Count missing values per column
missingValues = df.isnull()

print("Total missing values:", missingValues.sum())