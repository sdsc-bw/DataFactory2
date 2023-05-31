import copy
import pandas as pd

def convert_column_comma_and_set_type_float(col: pd.Series,) -> pd.Series:
    """
    Konvertiert Kommazahlen einer Spalte in die englische Schreibweise mit Punkt statt Komma.
    """
    col = col.map(lambda x: x.replace('.', '0.0').replace(',', '.') if type(x) != float else x).astype(float)
    return col

def convert_data_comma_and_set_type_float(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Konvertiert Kommazahlen eines Dataframes in die englische Schreibweise mit Punkt statt Komma.
    """
    tmp = copy.deepcopy(data)
    for i in data.columns:
        try:
            tmp[i] = convert_column_comma_and_set_type_float(data[i])
        except:
            tmp.drop([i], axis=1)
            if verbose:
                print(f'column {i} is not numerical')
            
    return tmp

def delete_columns(df, cols):
    if type(cols) == str:
        cols = [cols]
        
    for col in cols:
        if col in list(df.columns):
            df = df.drop(columns=col)
    return df