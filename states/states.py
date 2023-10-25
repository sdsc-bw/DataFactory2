########### replace with real dataset ##########
import copy
import pandas as pd
import os
import pickle

from methods.util import remove_inf_columns


# be careful when expanding, you might also need to change save_state and load_state
save_dirs = ['./states/data_states', './states/page_states', './states/page_states/data_transformation']

# manage page states       
def load_states():
    # load operations
    save_file = 'operations.pkl' 
    save_path = os.path.join(save_dirs[-1], save_file)
    if not os.path.exists(save_path):
        operations = {new_df_name: []}
    else:
        with open(save_path, 'rb') as f:
            operations= pickle.load(f)
    
    # load cards
    save_file = 'cards.pkl' 
    save_path = os.path.join(save_dirs[-1], save_file)
    if not os.path.exists(save_path):
        cards = {new_df_name: []}
    else:
        with open(save_path, 'rb') as f:
            cards= pickle.load(f)   
            
    # load children
    save_file = 'children.pkl' 
    save_path = os.path.join(save_dirs[-1], save_file)
    if not os.path.exists(save_path):
        children = {new_df_name: []}
    else:
        with open(save_path, 'rb') as f:
            children= pickle.load(f)
    return operations, cards, children

def initial_computations(sep, save_dir='./states/data_states'):
    all_dfs = {}
    for file in os.listdir(save_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            dataset_name = filename.split('.')[0]
            all_dfs[dataset_name] = load_dataset(dataset_name, sep, save_dir=save_dir)
    return all_dfs

def save_states():
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # save operations
    save_file = 'operations.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)

    with open(save_path_pages, 'wb') as f:
        pickle.dump(ALL_OPERATIONS, f)

    # save cards
    save_file = 'cards.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)

    with open(save_path_pages, 'wb') as f:
        pickle.dump(ALL_CARDS, f)

    # save children
    save_file = 'children.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)

    with open(save_path_pages, 'wb') as f:
        pickle.dump(ALL_CHILDREN, f)

# manage datasets
def save_dataset(df, dataset_name, sep, index_min=None, index_max=None):
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    
    # select range
    if index_min is not None:
        df = df.iloc[index_min:]
        
    if index_max is not None:
        df = df.iloc[:index_max]
    
    # save dataset
    save_path = os.path.join(save_dirs[0], dataset_name) + ".csv"
    df.to_csv(save_path, sep=sep, index=False)    
    
def delete_dataset(dataset_name):
    # save dataset
    save_path = os.path.join(save_dirs[0], dataset_name) + ".csv"
    
    if os.path.exists(save_path):
        os.remove(save_path)
    
def save_dataset_states(all_datasets, all_ranges, save_dir='./states/page_states'):
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
    # save ranges
    save_file = 'ranges.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)

    with open(save_path_pages, 'wb') as f:
        pickle.dump(all_ranges, f)
        
    # save datasets
    save_file = 'data.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)

    with open(save_path_pages, 'wb') as f:
        pickle.dump(all_datasets, f)
    
def delete_dataset_states(save_dir='./states/page_states'):
            
    # delete ranges
    save_file = 'ranges.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)
        
    if os.path.exists(save_path_pages):
        os.remove(save_path_pages)
        
    # delete datasets
    save_file = 'data.pkl'
    save_path_pages = os.path.join(save_dirs[-1], save_file)

    if os.path.exists(save_path_pages):
        os.remove(save_path_pages) 
    
def load_dataset(dataset_name, sep, save_dir='./states/data_states', index=None):
    # load dataset
    save_path = os.path.join(save_dir, dataset_name) + ".csv"
    
    if index is None:
        df = pd.read_csv(save_path, sep=sep)
    elif index == 'auto':
        df = pd.read_csv(save_path, sep=sep).reset_index()
        cols = ['index_auto']
        cols.extend(df.columns[1:])
        df.columns = cols
    else:
        df = pd.read_csv(save_path, sep=sep, index_col=index).reset_index()
        
    df = remove_inf_columns(df)    
    return df

def load_dataset_states():
    # load ranges
    save_file = 'ranges.pkl' 
    save_path = os.path.join(save_dirs[-1], save_file)
    if not os.path.exists(save_path):
        all_ranges = {}
    else:
        with open(save_path, 'rb') as f:
            all_ranges= pickle.load(f)
    
    # load max ranges
    save_file = 'max_ranges.pkl' 
    save_path = os.path.join(save_dirs[-1], save_file)
    if not os.path.exists(save_path):
        all_max_ranges = {}
    else:
        with open(save_path, 'rb') as f:
            all_max_ranges= pickle.load(f)
            
    # load datasets
    save_file = 'data.pkl' 
    save_path = os.path.join(save_dirs[-1], save_file)
    if not os.path.exists(save_path):
        all_datasets = {}
    else:
        with open(save_path, 'rb') as f:
            all_datasets = pickle.load(f)
            
    return all_ranges, all_max_ranges, all_datasets

def check_existence(dataset_name, save_dir='./states/data_states'):
    save_path = os.path.join(save_dir, dataset_name) + ".csv"
    return os.path.exists(save_path)

    
    
# global variables to access states and datasets
IN_PROCESSING_DATASETNAME = 'inprocessing_data'
CLEANED_DATASETNAME = 'cleaned_data'

#ALL_OPERATIONS, ALL_CARDS, ALL_CHILDREN = load_states()






