# Import necessary libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objs as go
from dateutil import parser

# import data
from data import table_data

# import utility
from methods.util import is_float

ENCODING_STRATEGIES = ['One Hot Encoding', 'Label Encoding', 'Date Encoding', 'Already Numeric', 'Replace Value']

ENCODING_LINKS = ['https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.OneHotEncoder.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html', 'https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/', '', '']

ENCODING_DESCRIPTIONS = ['Read more', 'Read more', 'Read more', 'This method is for categorical features that already contain only numbers.', 'This method is for replace certain feature values with another feature value of the same feature.']

def apply_encoding(df, col, strategy, in_str='', out_str=''):
    df = df.copy(deep=True)
    if type(col) == str:
        col = [col]
    
    if strategy == ENCODING_STRATEGIES[0]:
        return apply_one_hot_encoder(df, col)
    elif strategy == ENCODING_STRATEGIES[1]:
        return apply_label_encoder(df, col)
    elif strategy == ENCODING_STRATEGIES[2]:
        return apply_date_encoder(df, col)
    elif strategy == ENCODING_STRATEGIES[3]:
        return apply_numeric(df, col)
    elif strategy == ENCODING_STRATEGIES[4]:
        return apply_replace_value(df, col, in_str, out_str)
    else:
        print(f"Unknown encoding strategy: {strategy}. Apply One Hot Encoding instead.")
        return apply_one_hot_encoder(df, col)

def apply_one_hot_encoder(df, cols):
    oe = OneHotEncoder(drop='first', handle_unknown='ignore')
    dat_categ_onehot = df[cols]
    df_onehot = pd.DataFrame(oe.fit_transform(dat_categ_onehot.astype(str)).toarray(), index=dat_categ_onehot.index,
                             columns=oe.get_feature_names_out(dat_categ_onehot.columns))

    # del old features
    df = df.drop(cols, axis = 1)

    # concat the onehot feature back to the table data
    df = pd.concat([df, df_onehot], axis = 1)
          
    return df
    
def apply_label_encoder(df, cols):      
    for i in cols:     
        original = df[i]
        mask = df[i].isnull()
        df[i] = LabelEncoder().fit_transform(df[i].astype(str))
        df[i] = df[i].where(~mask, original)
        df[i] = df[i].apply(lambda x: int(x) if str(x) != 'nan' else np.nan)          
    return df

def apply_date_encoder(df, date_cols):
    for col in date_cols:
        df[col] = df[col].apply(lambda x: parser.parse(x) if isinstance(x, str) else x)

        # Encode year linearly
        df[col + ' year'] = df[col].dt.year

        # Encode other components using sine and cosine functions
        components = ['month', 'day', 'hour', 'minute', 'second', 'microsecond']
        for comp in components:
            df[col + ' ' + comp + ' sin'] = np.sin(2 * math.pi * df[col].dt.__getattribute__(comp) / df[col].dt.__getattribute__(comp).max())
            df[col + ' ' + comp + ' cos'] = np.cos(2 * math.pi * df[col].dt.__getattribute__(comp) / df[col].dt.__getattribute__(comp).max())

    # Remove the original date columns
    df.drop(date_cols, axis=1, inplace=True)
    
    # Remove columns with NaN values
    df.dropna(axis=1, how='all', inplace=True)

    return df
        
    
def apply_numeric(df, cols):
    for i in cols:
        #try:
        print(df[i])
        if df[i].str.contains('.').any() or df[i].str.contains(',').any():
            df[i] = convert_column_comma_and_set_type_float(df[i])
        else:
            df[i] = pd.to_numeric(df[i])
        print(df[i])
        #except:
        #    continue
    return df
        
def convert_column_comma_and_set_type_float(col: pd.Series,) -> pd.Series:
    """
    Konvertiert Kommazahlen einer Spalte in die englische Schreibweise mit Punkt statt Komma.
    """
    col = col.map(lambda x: x.replace('.', '0.0').replace(',', '.') if type(x) != float else x).astype(float)
    return col

def apply_replace_value(df, cols, in_str, out_str):   
    for i in cols:
        df[i] = df[i].replace(in_str, out_str)
    
    return df