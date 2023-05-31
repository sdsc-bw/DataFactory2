# Import necessary libraries
import pandas as pd
import math

# import data
from data import table_data

NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
OVERVIEW_COLUMNS = ['features', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', '%NA', 'datatype', 'unique', 'top', 'freq']

def check_if_cleaned(df):
    """
    Checks if a dataframe contains any NaN values or categorical features.

    Args:
    - df: pandas dataframe

    Returns:
    - string: "No NaN values or categorical features found" if there are none, 
              otherwise a string with the list of features containing NaN values or categorical features
    """

    nan_cols = df.columns[df.isna().any()].tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    return not nan_cols and not cat_cols

def get_percentage_nan_total(df):
    nan_percentage = round(df.isna().sum().sum()/(df.shape[0] * df.shape[1])  * 100, 2)
    return f"{nan_percentage}%"

def get_memory_usage(df):
    size_bytes = df.memory_usage(deep=True).sum()
    
    if size_bytes == 0:
        return 0, "B"
    
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}" 

def get_percentage_numeric(df):
    num_percentage = round(len(df.select_dtypes(include=NUMERICS).columns) / df.shape[1] *  100, 2)
    return f"{num_percentage}%"

def get_num_numeric_categorical(df):
    num_cat = len(df.select_dtypes(include='object').columns)
    num_num = len(df.columns) - num_cat
    return num_num, num_cat

def get_num_nan(df):
    df_na = df.isna().sum().to_frame()
    df_na.columns = ["#NA"]
    df_na = df_na.reset_index()    
    return df_na
    

def analyse_df(df):  
    description_num = analyse_numeric_data(df)
    description_cat = analyse_categorical_data(df)
    
    description = pd.concat([description_num, description_cat], axis=0, ignore_index=True)
    
    description.fillna('-', inplace=True)
    
    return description

def analyse_numeric_data(df):
    # select only numeric data    
    cols = df.select_dtypes(include=NUMERICS).columns
    df = df[cols]
    
    if len(cols) == 0:
        return pd.DataFrame([], columns=['features', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', '%NA', 'datatype'])
    
    # describe data
    description = df.describe().T.reset_index()
    description = description.round(2)
    description['%NA'] = get_percentage_nan(df).values
    description['datatype'] = get_dtypes(df).values 
    
    # rename index column
    cols = list(description.columns)
    cols[0] = 'features'
    description.columns = cols
    return description

def analyse_categorical_data(df):
    # select only categorical data
    cols = df.select_dtypes(include='object').columns
    df = df[cols]
    
    if len(cols) == 0:
        return pd.DataFrame([], columns=['features', 'count', 'unique', 'top', 'freq', '%NA', 'datatype'])
    
    # describe data
    description = df.describe().T.reset_index()
    description['%NA'] = get_percentage_nan(df).values
    description['datatype'] = get_dtypes(df).values 
    
    # rename index column
    cols = list(description.columns)
    cols[0] = 'features'
    description.columns = cols
    return description
    
def get_percentage_nan(df):
    nan_count = df.isna().sum() / df.shape[0] * 100
    nan_count = nan_count.round(2)
    return nan_count

def get_dtypes(df):
    dtypes = df.dtypes
    dtypes = dtypes.astype(str).replace('object', 'categorical')
    return dtypes

def compute_plot(df, col_index, cols, value_min, value_max, reset_index=False):    
    if col_index is None:
        sel = df.index.map(lambda x: x >= value_min and x <= value_max)
    else:
        sel = df[col_index].map(lambda x: x >= value_min and x <= value_max)
    if reset_index:
        df = df.loc[sel, cols].reset_index()
    else:
        df = df.loc[sel, cols]
    return df
    

def analyse_correlation(df):
    # only use numeric data  
    cols = df.select_dtypes(include=NUMERICS).columns
    df = df[cols]
    
    corr = df.corr()
    
    return corr

def compute_scatter(df, col, value_min, value_max):
    # only use numeric data  
    cols = df.select_dtypes(include=NUMERICS).columns
    df = df[cols]
    
    # filter data
    sel = table_data.DF_RAW[col].map(lambda x: x >= value_min and x <= value_max)
    df = table_data.DF_RAW.loc[sel, table_data.DF_RAW.columns].reset_index()
    return df
    