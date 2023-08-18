import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from datetime import timedelta

from scipy import stats
from scipy import fftpack
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, filtfilt, savgol_filter

import pykalman

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pywt

import pandas as pd

from methods.util import compare_lists


#from methods.constants import *

TRANSFORMATIONS_TABLE_DATA = ['Normalization', 'Standardization', 'PCA', 'Discrete Wavelet Transformation', 'Discrete Fourier Transformation']

TRANSFORMATIONS_TS = ['Normalization', 'Standardization', 'PCA', 'Discrete Wavelet Transformation', 'Discrete Fourier Transformation', 'Shifting', 'Sliding Window', 'Differencing', 'Savitzky-Golay Filter', 'Kalman Filter']

WAVELETS = {'Daubechies Wavelet': 'db', 'Symlets Wavelet': 'sym', 'Coiflets wavelets': 'coif', 'Haar Wavelet': 'haar'}

WAVELET_MODES = {'Symmetric': 'symmetric', 'Antisymmetric': 'antisymmetric', 'Zero': 'zero', 'Reflect': 'reflect', 'Periodic': 'periodic', 'Smooth': 'smooth', 'Periodization': 'periodization'}

SLIDING_WINDOW_OPERATIONS = {'Sum': 'sum', 'Mean': 'mean', 'Median': 'median', 'Min': 'min', 'Max': 'max', 'Std': 'std'}



def apply_transformation_table_data(df, cols, method, params):
    df = df.copy(deep=True)
    if method == TRANSFORMATIONS_TABLE_DATA[0]:
        return apply_normalization(df, cols)
    elif method == TRANSFORMATIONS_TABLE_DATA[1]:
        return apply_standardization(df, cols)
    elif method == TRANSFORMATIONS_TABLE_DATA[2]:
        return apply_pca(df, cols, params)
    elif method == TRANSFORMATIONS_TABLE_DATA[3]:
        return apply_dwt(df, cols, params)
    elif method == TRANSFORMATIONS_TABLE_DATA[4]:
        return apply_dft(df, cols)
    else:
        print(f'Unkown transformation: {method}')
        
def apply_transformation_time_series(df, cols, method, params):
    df = df.copy(deep=True)
    if method == TRANSFORMATIONS_TS[0]:
        return apply_normalization(df, cols)
    elif method == TRANSFORMATIONS_TS[1]:
        return apply_standardization(df, cols)
    elif method == TRANSFORMATIONS_TS[2]:
        return apply_pca(df, cols, params)
    elif method == TRANSFORMATIONS_TS[3]:
        return apply_dwt(df, cols, params)
    elif method == TRANSFORMATIONS_TS[4]:
        return apply_dft(df, cols)
    elif method == TRANSFORMATIONS_TS[5]:
        return apply_shifting(df, cols, params)
    elif method == TRANSFORMATIONS_TS[6]:
        return apply_sliding_window(df, cols, params)
    elif method == TRANSFORMATIONS_TS[7]:
        return apply_differencing(df, cols, params)
    elif method == TRANSFORMATIONS_TS[8]:
        return apply_savitzky_golay_filter(df, cols, params)
    elif method == TRANSFORMATIONS_TS[9]:
        return apply_kalman_filter(df, cols)
    else:
        print(f'Unkown transformation: {method}')
    

def apply_normalization(df, cols=None):
    """
    Normalizes selected columns of a pandas dataframe by scaling their values to the range [0, 1].
    
    Args:
    - df: pandas dataframe to be normalized
    - cols: list of column names to be normalized (default: all columns)
    
    Returns:
    - normalized pandas dataframe
    """
    # If no columns are specified, normalize all columns
    if cols is None:
        cols = df.columns.tolist()
    
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Fit the scaler to the selected columns and transform them
    normalized_data = scaler.fit_transform(df[cols].values)
    
    # Convert the normalized data to a pandas dataframe
    normalized = pd.DataFrame(normalized_data, index=df.index, columns=cols)
    
    # Concatenate the normalized columns with the unselected columns
    unselected_columns = [col for col in df.columns if col not in cols]
    normalized_df = pd.concat([df[unselected_columns], normalized], axis=1)
    
    return normalized_df, cols

def apply_standardization(df, cols):
    """
    Standardizes selected columns of a pandas dataframe by scaling their values to have zero mean and unit variance.
    
    Args:
    - df: pandas dataframe to be standardized
    - cols: list of column names to be standardized (default: all columns)
    
    Returns:
    - standardized pandas dataframe
    """
    # If no columns are specified, standardize all columns
    if cols is None:
        cols = df.columns.tolist()
    
    # Create a StandardScaler object
    scaler = StandardScaler()
    
    # Fit the scaler to the selected columns and transform them
    standardized_data = scaler.fit_transform(df[cols].values)
    
    # Convert the standardized data to a pandas dataframe
    standardized = pd.DataFrame(standardized_data, index=df.index, columns=cols)
    
    # Concatenate the standardized columns with the unselected columns
    unselected_columns = [col for col in df.columns if col not in cols]
    standardized_df = pd.concat([df[unselected_columns], standardized], axis=1)
    
    return standardized_df, cols

def apply_pca(df, cols, params):
    """
    Applies Principal Component Analysis (PCA) on a pandas dataframe for the given columns.
    
    Args:
    - df: pandas dataframe
    - cols: list of column names to be included in the PCA (default: all columns)
    - n_components: number of components to keep (default: all components)
    
    Returns:
    - pandas dataframe with the PCA components and remaining columns
    """
    feature_name = params['feature_name']
    del params['feature_name']
    # If no columns are specified, use all columns
    if cols is None:
        cols = df.columns.tolist()

    # Fit and transform the PCA on the selected columns
    pca = PCA(**params)
    pca_data = pca.fit_transform(df[cols])

    # Convert the PCA data to a pandas dataframe
    new_cols = [f'{feature_name}_'+str(i+1) for i in range(pca_data.shape[1])]
    pca_df = pd.DataFrame(pca_data, columns=new_cols, index=df.index)

    # Add the remaining columns to the PCA dataframe
    remaining_columns = [col for col in df.columns if col not in cols]
    pca_df = pd.concat([pca_df, df[remaining_columns]], axis=1)

    # Return the PCA dataframe
    return pca_df, new_cols
    
def apply_dwt(df, cols, params):
    """
    Applies discrete wavelet transformation on a pandas dataframe.

    Args:
    - df: pandas dataframe to be transformed
    - cols: cols where it should be applied
    - params: wavelet function and other params to be used

    Returns:
    - transformed pandas dataframe
    """
    wavelet = params['wavelet']
    level = params['level']
    mode = params['mode']

    # Parse wavelet family and parameters
    if wavelet != list(WAVELETS.values())[3]:
        n = params['n']
        wavelet = wavelet + str(n)

    # Create wavelet object
    wavelet_obj = pywt.Wavelet(wavelet)

    # Apply DWT row-wise
    transformed = df[cols].apply(lambda row: pywt.wavedec(row, wavelet_obj, level=level, mode=mode), axis=1)

    # Flatten the list of arrays
    flattened_transform = []
    for row in transformed:
        flattened_transform.append([item for sublist in row for item in sublist])

    # Create new columns for each coefficient of the wavelet transform
    new_cols = []
    for i in range(len(flattened_transform[0])):
        col_name = f'{wavelet}_coeff_{i}_lvl{level}'
        transformed_col = [row[i] for row in flattened_transform]
        df[col_name] = transformed_col
        new_cols.append(col_name)

    # Drop the original columns
    df = df.drop(cols, axis=1)

    return df, new_cols

def apply_dft(df, cols):
    """
    Applies discrete Fourier transformation on a pandas dataframe.

    Args:
    - df: pandas dataframe to be transformed
    - cols: cols where it should be applied

    Returns:
    - transformed pandas dataframe
    """
    # Apply DFT row-wise
    df_tmp = df[cols].reset_index(drop=False)
    transformed = df_tmp.apply(lambda row: np.fft.rfft(row).real, axis=1)

    # Convert list of arrays to a dataframe
    transformed = pd.DataFrame(transformed.tolist(), index=df.index)

    # Rename the transformed columns
    new_cols = [col + '_dft' for col in cols]
    transformed.columns = new_cols

    # Add the remaining columns to the PCA dataframe
    remaining_columns = [col for col in df.columns if col not in cols]
    transformed = pd.concat([transformed, df[remaining_columns]], axis=1)

    return transformed, new_cols

def apply_savitzky_golay_filter(df, cols, params):
    """
    This function smooths a time series dataframe using the Savitzky-Golay filter.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The time series dataframe to be smoothed.
    cols : str
        The name of the column in the dataframe to be smoothed.
    params : dict
        The parameters for the operation.
        
    Returns:
    --------
    pandas DataFrame
        The smoothed time series dataframe.
    """
    window_size = params['window_size']
    polyorder = params['polyorder']
    for col in cols:
        # Get the time series data as a numpy array
        data = df[col].values

        # Apply the Savitzky-Golay filter to smooth the data
        smoothed_data = savgol_filter(data, window_size, polyorder)

        # Create a new dataframe with the smoothed data
        smoothed_df = pd.DataFrame({col: smoothed_data}, index=df.index)

        # Add all the other columns from the original dataframe to the new smoothed dataframe
        smoothed_df = pd.concat([smoothed_df, df.drop(col, axis=1)], axis=1)
    
    return smoothed_df, cols

def apply_kalman_filter(df, cols):
    """
    This function applies a Kalman filter to the specified columns of a time series dataframe.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The time series dataframe to be filtered.
    cols : list
        A list of column names to be filtered.
        
    Returns:
    --------
    pandas DataFrame
        The filtered time series dataframe.
    """
    
    # Initialize the Kalman filter
    kf = pykalman.KalmanFilter()
    
    # Create an empty dataframe to store the filtered data
    filtered_df = pd.DataFrame(index=df.index)
    
    # Iterate over each column to be filtered
    for col in cols:
        
        # Get the time series data as a numpy array
        data = df[col].values
        
        # Apply the Kalman filter to the data
        filtered_data, _ = kf.filter(data)
        
        # Add the filtered data to the filtered dataframe
        filtered_df[col] = filtered_data
        
    # Add all the other columns from the original dataframe to the new filtered dataframe
    filtered_df = pd.concat([filtered_df, df.drop(cols, axis=1)], axis=1)
    
    return filtered_df, cols

def apply_sliding_window(df, cols, params):
    """
    This function applies a sliding window to a time series dataframe with the specified window size and operations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The time series dataframe to be windowed.
    params : dict
        The parameters for the operation.
        
    Returns:
    --------
    pandas DataFrame
        The windowed dataframe.
    """
    window_size = params['window_size']
    operation = params['operation']
    
    # Initialize an empty dataframe to store the windowed data
    windowed_df = pd.DataFrame()
    
    # Iterate over each column in the dataframe
    for col in cols:
        
        # Apply the specified operations to the column using a rolling window
        if 'sum' == operation:
            windowed_df[f'{col}_sum'] = df[col].rolling(window_size).sum()
        elif 'mean' == operation:
            windowed_df[f'{col}_mean'] = df[col].rolling(window_size).mean()
        elif 'median' == operation:
            windowed_df[f'{col}_median'] = df[col].rolling(window_size).median()
        elif 'min' == operation:
            windowed_df[f'{col}_min'] = df[col].rolling(window_size).min()
        elif 'max' == operation:
            windowed_df[f'{col}_max'] = df[col].rolling(window_size).max()
        elif 'std' == operation:
            windowed_df[f'{col}_std'] = df[col].rolling(window_size).std()
        
    # Add the windowed data to the windowed dataframe
    new_cols = list(windowed_df.columns)
    windowed_df = pd.concat([windowed_df, df], axis=1)
    
    windowed_df = windowed_df.drop(cols, axis=1)

    return windowed_df, new_cols

def apply_differencing(df, cols, params):
    """
    This function applies a sliding window to a time series dataframe with the specified window size and operations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The time series dataframe to be windowed.
    params : dict
        The parameters for the operation.
        
    Returns:
    --------
    pandas DataFrame
        The windowed dataframe.
    """
    window_size = params['window_size']
    
    # Initialize an empty dataframe to store the windowed data
    windowed_df = pd.DataFrame()
    
    # Iterate over each column in the dataframe
    for col in cols:
        
        # Apply the specified operations to the column using a rolling window
        windowed_df[f'{col}_diff'] = df[col].diff(periods=window_size)
    
    # Add the windowed data to the windowed dataframe
    new_cols = list(windowed_df.columns)
    windowed_df = pd.concat([windowed_df, df], axis=1)
    
    windowed_df = windowed_df.drop(cols, axis=1)
    
    return windowed_df, new_cols

def apply_shifting(df, cols, params):
    """
    Shifts the specified columns of a time series dataframe by the specified number of steps and adds the shifted columns
    as new columns in the dataframe. If `multi_shift` is True, multiple shifts will be created between 1 and `shift_steps`.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The time series dataframe to be shifted.
    cols : list of str
        The list of column names to shift.
    params : dict
        The parameters for the operation.
        
    Returns:
    --------
    pandas DataFrame
        The shifted dataframe.
    """
    shift_steps = params['shift_steps']
    multi_shift = params['multi_shift']
    
    # Initialize a dictionary to store the shifted columns
    shifted_columns = {}
    
    # Loop through each column in the dataframe and create the requested shifts for the specified columns
    for col in df.columns:
        if col in cols:
            if multi_shift:
                for i in range(1, shift_steps+1):
                    shifted_col_name = col + '_shifted_' + str(i)
                    shifted_col = df[col].shift(i)
                    shifted_columns[shifted_col_name] = shifted_col
            else:
                shifted_col_name = col + '_shifted_' + str(shift_steps)
                shifted_col = df[col].shift(shift_steps)
                shifted_columns[shifted_col_name] = shifted_col
        shifted_columns[col] = df[col]
    
    # Create a new dataframe with the shifted columns added
    shifted_df = pd.DataFrame(shifted_columns, index=df.index)
    shifted_df = shifted_df[shift_steps:]
    
    # get new_cols
    new_cols = compare_lists(list(df.columns), list(shifted_df.columns))
    cols += new_cols
    cols = sorted(cols)
    
    return shifted_df, cols