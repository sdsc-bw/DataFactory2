import numpy as np

NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

FEATURE_ENGINEERING_METHODS_TS = ['None', 'Resampling', 'FFT', 'Filtering', 'Transformation']
FEATURE_ENGINEERING_METHODS_COMPLETE_DF_TS = ['Resampling', 'FFT']

RESAMPLING_STRATEGIES_TS= ['mean', 'min', 'max', 'std']
FILTERING_STRATEGIES_TS= ['low pass', 'high pass', 'band pass']
TRANSFORMATION_STRATEGIES_TS= ['shift', 'sliding window', 'differencing', 'standardization', 'normalization']
SLIDING_WINDOW_STRATEGIES_TS= ['sum', 'mean', 'min', 'max', 'std']

SAMPLINGRATE_OPTIONS_TS = {'Day(s)': 'days', 'Hour(s)': 'hours', 'Minute(s)': 'minutes', 'Second(s)': 'seconds'}
TIMESTAMP_TO_INDEX = {'days': 60*60*24, 'hours': 60*60, 'minutes': 60, 'seconds': 1} # used, when index type is not datetime

def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
    
def remove_item_if_exist(l: list, item):
    if item in l:
        l.remove(item)
        
        
def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def count_unique_values(df, col):
    counts = df[col].value_counts()
    counts = counts[:15]
    return counts

def get_nan_positions(df, cols):
    if type(cols) == str:
        cols = [cols]
    
    nan_positions = []
    for col in cols:
        col_nan_positions = list(np.where(df[col].isnull())[0])
        nan_positions.append(col_nan_positions)
    return nan_positions

def compare_lists(list1, list2):
    """
    Compares the elements of two lists and returns a list containing the elements that are unique to each list.
    
    Parameters:
    -----------
    list1 : list
        The first list to compare.
    list2 : list
        The second list to compare.
        
    Returns:
    --------
    list
        A list containing the elements that are unique to `list1` or `list2`.
    """
    set1 = set(list1)
    set2 = set(list2)
    
    unique_to_list1 = list(set1.difference(set2))
    unique_to_list2 = list(set2.difference(set1))
    
    return unique_to_list1 + unique_to_list2

def remove_inf_columns(df):
    cols = []
    for column in df.select_dtypes(include=NUMERICS).columns:
        if np.any(np.isinf(df[column])) or np.any(np.abs(df[column]) > np.finfo(np.float64).max):
            cols.append(column)
            
    df = df.drop(cols, axis=1)
    
    return df
     

