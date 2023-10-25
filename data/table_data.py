import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype

# the load data and set first column to index with name
import os

# import state management
from states.states import *


SEP = ','
index = 'auto'

DF_RAW = None

def find_problematic_columns(dataframe):
    cols = []
    for column in dataframe.columns:
        if np.any(np.isinf(dataframe[column])) or np.any(np.abs(dataframe[column]) > np.finfo(np.float64).max):
            cols.append(column)
    return cols


ALL_RANGES, ALL_MAX_RANGES, ALL_DATASETS = load_dataset_states() 
ALL_RESULTS_CLASSIFICATION = []
        
