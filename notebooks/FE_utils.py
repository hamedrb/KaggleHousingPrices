## Feature engineering utils

import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

##########  Categorical variable encoding  ##########################

def custom_combiner(feature, category):
    ## Redefine the feature_name_combiner of OneHotEncoder
    return str(feature) + "__" + str(category)

def one_hot_encoding(df, col, encoder=None):
    """
    One-hot encoding of a column of a dataframe
    
    Parameters
    -------------------
    df: DataFrame
    col: str
    encoder: OneHotEncoder or None
        One-hot encoder already fitted. Used in case of application of the encoder
        on test data.
    
    Return
    -------------------
    df: DataFrame
    encoder: OneHotEncoder
    """
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', 
                                feature_name_combiner=custom_combiner, 
                                sparse_output=False,
                                drop='if_binary')
        transformed_array = encoder.fit_transform(df[col].to_numpy().reshape(-1, 1))
    else:
        transformed_array = encoder.transform(df[col].to_numpy().reshape(-1, 1))
    
    transformed_df = pd.DataFrame(transformed_array, 
                                  columns=encoder.get_feature_names_out([col]), 
                                  index=df.index)
    df = pd.concat([df, transformed_df], axis=1).drop([col], axis=1)
        
    return df, encoder

def ordinal_encoding(df, col, categories='auto', encoder=None):
    """
    Ordinal encoding of a column of a dataframe
    
    Parameters
    -------------------
    df: DataFrame
    col: str
    categories: list
        List of categories to be converted to 0, 1, 2, ...
    encoder: OrdinalEncoder or None
        Ordinal encoder already fitted. Used in case of application of the encoder
        on test data.
    Return
    -------------------
    df: DataFrame
    encoder: OrdinalEncoder
    """
    if categories!='auto':
        categories = [categories]
        
    if encoder is None: 
        encoder = OrdinalEncoder(categories=categories)
        transformed_array = encoder.fit_transform(df[col].to_numpy().reshape(-1, 1))
    else:
        transformed_array = encoder.transform(df[col].to_numpy().reshape(-1, 1))
    
    transformed_df = pd.DataFrame(transformed_array, 
                                  columns=[col], 
                                  index=df.index)
    df[col] = transformed_df[col]
    
    return df, encoder

##########  Read data  ##########################

def read_params(config_path):
    ## Read config file
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

##########  Graphs  #############################

def hist_boxplot(df, col):
    """
    Plot a histogram of a continuous variable with a boxplot
    on top of it.
    
    Parameters
    -------------------
    df: DataFrame
    col: str
    	 The continuous variable.
    
    Return
    -------------------
    Nothing.
    """
    # creating a figure composed of two matplotlib.Axes objects
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    # assigning a graph to each ax
    sns.boxplot(df[col], ax=ax_box, orient='h', 
                flierprops={"marker": "x"},
                boxprops={"facecolor": (.4, .6, .8, .5)})
    sns.histplot(data=df, x=col, ax=ax_hist)
       
    # Remove x axis name and y axis ticks for the boxplot
    ax_box.set(xlabel='')
    ax_box.set(yticks=[])
    plt.show()

##########  Outliers  #############################

def outlier_limits(df, col):
    """
    Give the outlier limits of a continuous variable detected by IQR method.
    
    Parameters
    -------------------
    df: DataFrame
    col: str
    	 The continuous variable.
    
    Return
    -------------------
    lower_limit: float
	upper_limit: float
    """
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3-q1
    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr
    return lower_limit, upper_limit


def n_outliers(df, col):
    """
    Give the number of outliers of a continuous variable detected by IQR method.
    
    Parameters
    -------------------
    df: DataFrame
    col: str
    	 The continuous variable.
    
    Return
    -------------------
    n_low_outliers: int
	    Number of low outliers.
	n_high_outliers: int
        Number of high outliers.
    """
    lower_limit, upper_limit = outlier_limits(df, col)
    n_low_outliers = len(df[df[col]<lower_limit])
    n_high_outliers = len(df[df[col]>upper_limit])
    return n_low_outliers, n_high_outliers


def replace_outliers_with_extreme(df, col, low=False, high=False):
    """
    Replace low and/or high outliers with minimum and/or maximum non-outliers respectively.
    
    Parameters
    -------------------
    df: DataFrame
    col: str
    	 The continuous variable.
    low: bool
         If low==True, replace low outliers.
    high: bool
         If high==True, replace high outliers.
    
    Return
    -------------------
    df: DataFrame
    """
    lower_limit, upper_limit = outlier_limits(df, col)
    if low:
        df.loc[df[col]<lower_limit, col] = lower_limit
    if high:
        df.loc[df[col]>upper_limit, col] = upper_limit
    return df
	
