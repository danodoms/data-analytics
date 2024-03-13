# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 20:25:44 2024

@author: admin
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

selected_dataset = pd.read_csv('rice-yield-act3.csv')
# selected_dataset = pd.read_csv('dataset2.csv')

df = selected_dataset





def encode_dataframe(df):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:
    df (DataFrame): The input DataFrame containing categorical columns to be encoded.

    Returns:
    encoded_df (DataFrame): A new DataFrame with categorical columns encoded.
    """
    encoded_df = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    
    # Iterate through each column in the DataFrame
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    
    return encoded_df




df_encoded = encode_dataframe(df)



def preserve_no_nan_columns(df):
    """
    Create a duplicate DataFrame preserving columns with no NaN values and only the first column with NaN.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    new_df (DataFrame): A new DataFrame with columns containing no NaN values and only the first column with NaN.
    """
    nan_detected = False
    columns_to_keep = []
    
    print("columns to check ", df_encoded.columns)
    
    for column in df_encoded.columns:
        print("now checking column ", column)

        if not df_encoded[column].isna().any():
            columns_to_keep.append(column)
                
        elif df_encoded[column].isna().any() and nan_detected:
            continue
        
        elif df_encoded[column].isna().any() and not nan_detected:
            nan_detected = True
            columns_to_keep.append(column)
        
    
    new_df = df_encoded[columns_to_keep].copy()
    return new_df







def get_nan_column(df):
    for column in df.columns:
        if df[column].isna().any():
            return column
    return False
            
        

count = 1

while(get_nan_column(df)):
    
    print('////////////////////////////////')
    print('/////// ENTERED WHILE LOOP /////')
    print('/////// PHASE', count, ' //////')
    count+=1
    
    print('df has nan column: ', get_nan_column(df))
    
    df_for_model = preserve_no_nan_columns(df_encoded)
    column_to_predict = get_nan_column(df_for_model)
    
    
    train_df = df_for_model.dropna(axis=0)
    # test_df = df_for_model[df.isnull().any(axis=1)]
    test_df = df_for_model[df_for_model[column_to_predict].isnull()]
    
    
    haha = test_df.loc[test_df[column_to_predict].isnull(), column_to_predict]
    
    
    x_train = train_df.drop(column_to_predict, axis=1)
    y_train = train_df[column_to_predict]
    
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    x_test = test_df.drop(column_to_predict, axis=1)
    y_pred = lr.predict(x_test)
    
    test_df[column_to_predict] = y_pred
    # test_df.loc[test_df[column_to_predict].isnull(), column_to_predict] = y_pred
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    # df = pd.merge(train_df, test_df, how="left")
    # df.sort_index(inplace=True)
    df.to_csv('trial.csv')





    
