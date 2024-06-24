import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.plotting import heatmap
from wordcloud import WordCloud
import openpyxl

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

import os

#------------------------------------------------------------------------    
# 1. ES - CLEANING FUNCTION
# erik 
def clean_es_features(train, colors): 
    #Type - transforming all 2's to 0's
    train['Type'] = train['Type'].replace(2, 0) 
    #has_Video - transforming all non 0's to 1's
    train['has_Video'] = (train['VideoAmt'] != 0).astype(int)
    #has_Photo - transforming all non 0's to 1's
    train['has_Photo'] = (train['PhotoAmt'] != 0).astype(int)
    #MaturitySize - replacing all 0's with -1's
    train['MaturitySize'] = train['MaturitySize'].replace(0, -1)
    #Maturity_isSpecified
    train['Maturity_isSpecified'] = (train['MaturitySize'] != 0).astype(int)
    #FurLength - replacing all 0's with -1's
    train['FurLength'] = train['FurLength'].replace(0, -1)
    #FurLength_isSpecified
    train['FurLength_isSpecified'] = (train['FurLength'] != 0).astype(int)
    #isMale - transform to binary
    train['isMale'] = train['Gender'].apply(lambda x: 1 if x == 1 or x == 3 else 0) 
    #isFemale - transform to binary
    train['isFemale'] = train['Gender'].apply(lambda x: 1 if x == 2 or x == 3 else 0) 
    #{Color} - OHE for presence of each color
    for color_num, color in zip(colors['ColorID'], colors['ColorName']):
        train[color] = train[['Color1', 'Color2', 'Color3']].apply(lambda row: 1 if color_num in row.values else 0, axis=1) 
    #ColorCount
    color_columns = colors['ColorName'].tolist()
    train['ColorCount'] = train[color_columns].sum(axis=1)
    return train


#------------------------------------------------------------------------    
# 2.1 ALR - HELPER FUNCTIONS 


# Helper function to plot countplot
def plot_breakdown_adoption(df, target, hue='AdoptionSpeed', title=''):
    # Check if column_name exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the DataFrame.")
    # Plot
    graph = sns.countplot(x=target, data=df, hue=hue)
    plt.title(f'AdoptionSpeed by {title}');

# Define function to provide % split Adoption within each target condition 
def table_breakdown_adoption(df, target):
    # Check if column_name exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the DataFrame.")   
    # Create count by AdoptionSpeed and target col
    percentage_df = df.groupby([target, 'AdoptionSpeed']).size().reset_index(name='count') 
    # Obtain percentages
    percentage_df['percentage'] = percentage_df.groupby(target, group_keys=False)['count'].apply(lambda x: (x / x.sum()) * 100)
    percentage_df['percentage'] = round(percentage_df['percentage'],1)   
    # Sort result
    percentage_df = percentage_df.sort_values([target, 'AdoptionSpeed'])
    return percentage_df

# Define function to OHE
def OHE_vars(df, cols):     
    df_selected = df[cols]
    # Creating an instance of the OneHotEncoder
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    # Encoding the selected columns
    encoded_columns = encoder.fit_transform(df_selected)
    # Creating new column names based on the categories
    new_column_names = encoder.get_feature_names_out(cols)
    # Creating a new DataFrame with the encoded columns
    df_encoded = pd.DataFrame(encoded_columns, columns=new_column_names)
    # Combining the encoded DataFrame with the original DataFrame
    df = pd.concat([df, df_encoded], axis=1)
    #df = pd.concat([df.drop(cols, axis=1), df_encoded], axis=1)
    return df

# Binary Encoding
def binary_encoding(df, target):
    # Check if column_name exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the DataFrame.")
    # Encode
    temp_encoder = ce.BinaryEncoder(cols=[target])
    df = temp_encoder.fit_transform(df)
    return df 
   # Relabel numeric values to [0,2] instead of [1,3]

# Relabel numeric values to [0,2] instead of [1,3]
def relabel_col(df, target, to_range, from_range):
    # Check if column_name exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Column '{target}' does not exist in the DataFrame.")       
    # Create dict and apply function
    dict_lab = {i: j for i, j in zip(to_range, from_range)}
    df[target] = df[target].map(dict_lab)
    return df

# Normalizing function 
def normalize_var(df, target):
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    # Select column and reshape it to a 2-dimensional array
    target_values = df[target].values.reshape(-1, 1)
    # Fit and transform the 'Fee' values using StandardScaler
    target_scaled = scaler.fit_transform(target_values)
    # Replace the 'Fee' column in the DataFrame with the scaled values
    df[target] = target_scaled
    return df


# -------------------
# # 2.2 ALR - COMPREHENSIVE CLEANING FUNCTION (INTEGRATES HELPER FUNCTIONS ABOVE):  

# # def clean_alr_features(train, states): 
#     # Identify and process invalid names :
#     ## Fill NAs
#     train['Name'].fillna('No_Name', inplace=True)
    


#     ## Create new column to identify invalid_names
#     invalid_name(train, 'Name') # Apply function to our dataframe
    

#     # relabel 'Health' column

    
#     relabel_col(train,'Health',range(1,4), range(0,3))
#     # create dewormed/vaccinated string column for reference
#     train['dew_vacc'] = train['Dewormed'].astype(str)+ 'D+' +train['Vaccinated'].astype(str)+ 'V'
#     # Run OHE on Vaccinated, Dewormed, Sterilized (3 columns)
#     selected_columns1 = ['Vaccinated', 'Dewormed','Sterilized']
#     train = OHE_vars(train, selected_columns1)
    
#     # create state dictionary for mapping State names to State ids
#     state_dict = { i:j for i,j in zip(states['StateID'], states['StateName']) } # WHERE TO INTEGRATE THIS?
#     train['State'] = train['State'].map(state_dict)
   
#     # apply binary encoding function to 'State' col
#     train = binary_encoding(train, 'State')
#     # Create binary 'fee' variable if pets is Free or not
#     train['Fee_binary'] = (train['Fee'] > 1).astype(int)
#     # Clipping Fee values between $0-400 (remove outliers)
#     train['Fee'] = train['Fee'].clip(0, 400)
    

        
#     # Apply normalization to clipped 'Fee' variable
#     train = normalize_var(train, 'Fee')
    
#     # Create new binary 'Quantity' feature (single or multiple pets in listing)
#     train['Quantity_binary'] = (train['Quantity'] > 1).astype(int)
#     # create 'Age_guessed' var (if Age='guessed')
#     train['Age_guessed'] = train['Age'].apply(lambda x: 1 if x in range(12,12*10, 12) else 0)

# ------------------------------------------------------------------------------------------
# IMAGE METADATA (BAILEY) : 
def get_variables(file_id):
    """
    """
    
    # check if the file exists
    try:
        # Make a dictionary from the json file
        
        with open(f'../Dataset/test_metadata/{file_id}.json') as f:
            dic = json.load(f)
        
        # Close the file
        f.close()
                
    # return None if the file doesn't exist (go to next PetID)
    except:
        return None 
    
    # Hard code keys because they're all the same
    # Create df to get label annotations data
    df = pd.DataFrame.from_dict(dic['labelAnnotations'])
    
    # Create df to get image properties data
    df2 = pd.DataFrame.from_dict(dic['imagePropertiesAnnotation']['dominantColors']['colors'])
    
    # Merge the two dfs
    merged_df = pd.merge(df,df2, left_index=True, right_index=True)

    # Return the merged df containing all data for the image (taking the highest confidence row)  
    ret = merged_df.iloc[0]
    
    return ret

# ------------------------------------------------------------------------------------------



    
    