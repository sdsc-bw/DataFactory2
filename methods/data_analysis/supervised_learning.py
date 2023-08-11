import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


CLASSFIER = ["Baseline", "KNN", "Random Forest", "Gradient Boosting"]
REGRESSOR = ["Baseline", "Linear", "Random Forest", "Gradient Boosting"]

CLASSIFIER_BASELINE_STRATEGY = {"Prior": "prior", "Most Frequent": "most_frequent", "Stratified": "stratified", "Constant": "constant"}
REGRESSOR_BASELINE_STRATEGY = {"Mean": "mean", "Median": "median", "Quantile": "quantile", "Constant": "constant"}

CLASSIFIER_KNN_ALGORITHM = {"Auto": "auto", "Ball Tree": "ball_tree", "KD Tree": "kd_tree", "Brute": "brute"}
CLASSIFIER_KNN_WEIGHTS =  {'Uniform': 'uniform', 'Distance': 'distance'}

CLASSIFIER_RF_CRITERION = {"Gini": "gini", "Entropy": "entropy", "Log Loss": "log_loss"}
REGRESSOR_RF_CRITERION = {"Squared Error": "squared_error", "Absolute error": "absolute_error", "Friedman MSE": "friedman_mse", "Poisson": "poisson"}

def apply_classifier(df, target, train_size, model, params, ts_cross_val=False):
    # Extract the target values
    y = df[target].values

    # Extract the feature values (excluding the target column)
    X = df.drop(columns=[target]).values
    
    # compute number of folds
    cv = int(1.0 / (1 - train_size))
    if cv < 2:
        cv = 2
    
    # initialize classifier
    if model == CLASSFIER[0]:
        clf = DummyClassifier(**params)
    elif model == CLASSFIER[1]:
        clf = KNeighborsClassifier(**params)
    elif model == CLASSFIER[2]:
        clf = RandomForestClassifier(**params)        
    elif model == CLASSFIER[3]:
        clf = XGBClassifier(**params)
    else:
        print(f"Unknown model: {model}")
        
    # evalutate classifier
    if ts_cross_val:
        scores = time_series_cross_val(X, y, cv, clf, scoring='f1_macro')
    else: 
        scores = cross_val(X, y, cv, clf, scoring='f1_macro')
        
    if scores.isna().any().any():
        raise ValueError("Target contains contains continous values. Please check your target.")
    
    return scores

def apply_regressor(df, target, train_size, model, params, ts_cross_val=False):
    # Extract the target values
    y = df[target].values

    # Extract the feature values (excluding the target column)
    X = df.drop(columns=[target]).values
    
    # compute number of folds
    cv = int(1.0 / (1 - train_size))
    if cv < 2:
        cv = 2
    
    # initialize classifier
    if model == REGRESSOR[0]:
        reg = DummyRegressor(**params)
    elif model == REGRESSOR[1]:
        reg = LinearRegression(**params)
    elif model == REGRESSOR[2]:
        reg = RandomForestRegressor(**params)        
    elif model == REGRESSOR[3]:
        reg = XGBRegressor(**params)
    else:
        print(f"Unknown model: {model}")
        
    # evalutate classifier
    if ts_cross_val:
        scores = time_series_cross_val(X, y, cv, reg, scoring='neg_mean_squared_error')
    else: 
        scores = cross_val(X, y, cv, reg, scoring='neg_mean_squared_error')
        
    if scores.isna().any().any():
        raise ValueError("Target contains contains continous values. Please check your target.")
    
    return scores

def cross_val(X, y, cv, model, scoring):

    # Perform cross-validation and collect scores
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Convert negative mean squared errors to positive 
    if scoring == 'neg_mean_squared_error':
        scores = -scores

    # Create a DataFrame with fold indices and scores
    df_scores = pd.DataFrame({'Fold': range(1, len(scores) + 1), 'Score': scores})

    return df_scores

def time_series_cross_val(X, y, cv, model, scoring):
    # Create TimeSeriesSplit cross-validation generator
    tscv = TimeSeriesSplit(n_splits=cv)

    # Perform cross-validation and collect scores
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

    # Convert negative mean squared errors to positive 
    if scoring == 'neg_mean_squared_error':
        scores = -scores

    # Create a DataFrame with fold indices and scores
    df_scores = pd.DataFrame({'Fold': range(1, len(scores) + 1), 'Score': scores})

    return df_scores
    
        