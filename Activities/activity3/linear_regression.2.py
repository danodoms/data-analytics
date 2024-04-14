"""
Created on Thu Feb 29 20:25:44 2024

@author: admin
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression



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
    
    print("columns to check ", df.columns)
    
    for column in df.columns:
        print("now checking column ", column)

        if not df[column].isna().any():
            columns_to_keep.append(column)
                
        elif df[column].isna().any() and nan_detected:
            continue
        
        elif df[column].isna().any() and not nan_detected:
            nan_detected = True
            columns_to_keep.append(column)
        
    new_df = df[columns_to_keep].copy()
    return new_df



def get_nan_column(df):
    for column in df.columns:
        if df[column].isna().any():
            return column
    return False
            
# selected_dataset = pd.read_csv('rice-yield-act3.csv')
selected_dataset = pd.read_csv('banana_disease_data_numerical.csv')

# selected_dataset = pd.read_csv('dataset2.csv')

# stores the changes of each process
df = encode_dataframe(selected_dataset)        

# duplicate dataframe that is used for processing
df_copy = df


while(get_nan_column(df)): # use original df
    print('//////////////////////////////////////')
    print('nan column: ', get_nan_column(df)) # use original df
    print('//////////////////////////////////////')
    
    # df_for_model = preserve_no_nan_columns(df) # dataframe with only one nan column, use original df
    df_for_model = df[['Environmental Conditions', 'Geographical Location', 'Cultural Practices', 'Disease History', 'Disease Surveillance and Monitoring']]
    column_to_predict = get_nan_column(df_for_model) 
    
    
    # Separate to train and test
    train_df = df_for_model.dropna(axis=0)
    test_df = df_for_model[df_for_model[column_to_predict].isnull()]
    
    
    # Create x and y train
    x_train = train_df.drop(column_to_predict, axis=1)
    y_train = train_df[column_to_predict]
    
    
    # Create x test
    x_test = test_df.drop(column_to_predict, axis=1)
    
    
    # Create the model
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    
    # Apply model
    y_pred = lr.predict(x_test)
    
    test_df[column_to_predict] = y_pred
    
    df_imputed = train_df.add(test_df, axis=1, fill_value=0)
    df[column_to_predict] = df_imputed[column_to_predict]
    

df.to_csv('trial_output_banana.csv')


# # Standardization
# from sklearn.preprocessing import StandardScaler
# scalerSD = StandardScaler()
# df_standardized = pd.DataFrame(scalerSD.fit_transform(df), columns=df.columns)
# df_standardized.to_csv('output-std.csv', index=False)


# # Normalization
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df_normalized = pd.DataFrame(scaler.fit_transform(df_standardized), columns=df_standardized.columns)
# df_normalized.to_csv('output-std-norm.csv', index=False)




    
