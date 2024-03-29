# Import necessary libraries
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import pandas as pd

# import utility
from methods.util import is_float

IMPUTER_METHODS = ['Simple Imputer', 'Iterative Imputer', 'KNN Imputer', 'Manual']
IMPUTER_STRATEGIES = {'Mean': 'mean', 'Median': 'median', 'Most Frequent': 'most_frequent', 'Constant': 'constant'}
IMPUTER_ORDER = {'Ascending': 'ascending', 'Descending': 'descending', 'Roman': 'roman', 'Arabic': 'arabic', 'Random': 'random'}
IMPUTER_WEIGHTS =  {'Uniform': 'uniform', 'Distance': 'distance'}

IMPUTER_LINKS = ['https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html', '']
IMPUTER_DESCRIPTIONS = ['Read more', 'Read more', 'Read more', 'Allows to change the feature value of a certain index.']
        
def apply_imputing(df, cols, method, params):
    df = df.copy(deep=True)
    
    if type(cols) == str:
        cols = [cols]
    
    if method == IMPUTER_METHODS[0]:
        return apply_simple_imputer(df, cols, params)
    elif method == IMPUTER_METHODS[1]:
        return apply_iterative_imputer(df, cols, params)
    elif method == IMPUTER_METHODS[2]:
        return apply_knn_imputer(df, cols, params)
    elif method == IMPUTER_METHODS[3]:
        return apply_manual(df, cols, params)
    else:
        print(f'Unknown imputing method: {method}')        
        
        
def apply_simple_imputer(df, cols, params):
    imputer = SimpleImputer(**params)
    
    df = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)
        
    return df
        
def apply_iterative_imputer(df, cols, params):
    imputer = IterativeImputer(**params)
    
    transformed_data = imputer.fit_transform(df[cols])
    
    df = pd.DataFrame(transformed_data[:, 0], columns=[cols[0]], index=df.index)
    
    return df
        
def apply_knn_imputer(df, cols, params):
    imputer = KNNImputer(**params)
    
    transformed_data = imputer.fit_transform(df[cols])
    
    df = pd.DataFrame(transformed_data[:, 0], columns=[cols[0]], index=df.index)
    
    return df
    
def apply_manual(df, cols, params):
    index = params['index']
    fill_value = params['fill_value']
    for i in cols:
        df[i].iloc[index] = fill_value
    return df