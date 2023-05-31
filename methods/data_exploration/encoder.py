# Import necessary libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# import data
from data import table_data

# import utility
from methods.util import is_float

ENCODING_STRATEGIES = ['One Hot Encoding', 'Label Encoding', 'Already Numeric', 'Replace Value']

def apply_encoding(df, col, strategy, in_str='', out_str=''):
    df = df.copy(deep=True)
    if type(col) == str:
        col = [col]
    
    if strategy == ENCODING_STRATEGIES[0]:
        return apply_one_hot_encoder(df, col)
    elif strategy == ENCODING_STRATEGIES[1]:
        return apply_label_encoder(df, col)
    elif strategy == ENCODING_STRATEGIES[2]:
        return apply_numeric(df, col)
    elif strategy == ENCODING_STRATEGIES[3]:
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
        print(df[cols])
        print(df[i].dtypes)
        df[i] = df[i].where(~mask, original)
        df[i] = df[i].apply(lambda x: int(x) if str(x) != 'nan' else np.nan)          
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