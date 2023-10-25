import pandas as pd
import numpy as np

from scipy.stats import iqr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

OUTLIER_DETECTION_METHODS = ["Isolation Forest", "Local Outlier Detector", "KV Detector"]
OUTLIER_DETECTION_LOCAL_ALGORITHM = {"Auto": "auto", "Ball Tree": "ball_tree", "KD Tree": "kd_tree", "Brute": "brute"}
OUTLIER_DETECTION_LOCAL_OUTLIER_FACTOR_METRIC = {'Minkowski': 'minkowski', 'Euclidean': 'euclidean', 'Manhattan': 'manhattan', 'Chebyshev': 'chebyshev', 'Mahalanobis': 'mahalanobis'}

OUTLIER_LINKS = ['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html', '']
OUTLIER_DESCRIPTIONS = ['Read more', 'Read more', 'This method is using the interquartile range (IQR) and mean. Datapoints that are more than 3 times the IQR away from the mean in either direction are considered outliers.']

def apply_outlier_detection(df, method, params):
    if method == OUTLIER_DETECTION_METHODS[0]:
        return apply_isolation_forest(df, params) 
    elif method == OUTLIER_DETECTION_METHODS[1]:
        return apply_local_outlier_factor(df, params) 
    elif method == OUTLIER_DETECTION_METHODS[2]:
        return apply_kv_detector(df, params)
    else:
        print(f'Unknown outlier method: {method}') 

def apply_isolation_forest(df, params):

    clf = IsolationForest(**params)
    out = pd.Series(clf.fit_predict(df), index=df.index)
    is_outlier = out.map(lambda x: x == -1)
    

    # PCA
    if len(df.columns) > 2:
        df_tmp = apply_pca(df)
    elif len(df.columns) > 1:
        df_tmp = df.copy(deep=True)
        df_tmp = ["x", "y"]
    else:
        df_tmp = pd.DataFrame([])
        df_tmp['x'] = df.index
        df_tmp['y'] = df[df.columns[0]]
    df_tmp["Is Outlier"] = is_outlier
    return df_tmp, is_outlier

def apply_local_outlier_factor(df, params):
    
    # detection
    clf = LocalOutlierFactor(**params)
    out = pd.Series(clf.fit_predict(df), index = df.index)
    is_outlier = out.map(lambda x: x == -1)

    # PCA
    if len(df.columns) > 2:
        df_tmp = apply_pca(df)
    elif len(df.columns) > 1:
        df_tmp = df.copy(deep=True)
        df_tmp = ["x", "y"]
    else:
        df_tmp = pd.DataFrame([])
        df_tmp['x'] = df.index
        df_tmp['y'] = df[df.columns[0]]
    df_tmp["Is Outlier"] = is_outlier
    return df_tmp, is_outlier

def apply_kv_detector(df, params):
    col = params['feature']
    series_num = df[col]

    # detection
    v_iqr = iqr(series_num)
    v_mean = series_num.mean()
    ceiling = v_mean + 3*v_iqr
    floor = v_mean - 3*v_iqr
    is_outlier = series_num.map(lambda x: x>ceiling or x<floor)

    df_tmp = pd.DataFrame([])
    df_tmp['x'] = df.index
    df_tmp['y'] = df[col]
    df_tmp["Is Outlier"] = is_outlier
    return df_tmp, is_outlier

def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    df_tmp = pd.DataFrame(pca.fit_transform(df), columns = ["x", "y"])
    return df_tmp