import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype
import sqlite3
import mysql.connector
import psycopg2
import pickle

# the load data and set first column to index with name
import os

# import state management
from states.states import *


SEP = ','
index = 'auto'

if not check_existence(IN_PROCESSING_DATASETNAME):
    DF_RAW = None
else:
    index = None
    DF_RAW = load_dataset(IN_PROCESSING_DATASETNAME, SEP, index=index)

def find_problematic_columns(dataframe):
    cols = []
    for column in dataframe.columns:
        if np.any(np.isinf(dataframe[column])) or np.any(np.abs(dataframe[column]) > np.finfo(np.float64).max):
            cols.append(column)
    return cols


ALL_RANGES, ALL_DATASETS = load_dataset_states() 
ALL_RESULTS_CLASSIFICATION = []
        
