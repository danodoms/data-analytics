#import lib
import pandas as pd
import numpy as np

#load dataset
dataset = pd.read_csv('activity1.csv')
print(dataset.describe())

#create dependent & independent vars
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)

#count number of missing values in each column
print(dataset.isnull().sum())

#replace missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])