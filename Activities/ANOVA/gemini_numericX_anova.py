#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 07:19:15 2024

@author: danodoms
"""

import pandas as pd
import statsmodels.api as sm

# Example dataset (replace with your actual data)
df = pd.read_csv("iris.csv")

df.head()


# # One-way ANOVA
# model = sm.OLS.from_formula('X ~ Y', data=df)
# result = model.fit()

# print(result.summary())

