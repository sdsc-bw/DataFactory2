import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.model_selection import train_test_split


CLASSFIER = ["Baseline", "KNN", "Random Forest", "Gradient Boosting"]
REGRESSOR = ["Baseline", "Linear", "Random Forest", "Gradient Boosting"]

CLASSIFIER_BASELINE_STRATEGY = {"Prior": "prior", "Most Frequent": "most_frequent", "Stratified": "stratified", "Constant": "constant"}
REGRESSOR_BASELINE_STRATEGY = {"Mean": "mean", "Median": "median", "Quantile": "quantile", "Constant": "constant"}

CLASSIFIER_KNN_ALGORITHM = {"Auto": "auto", "Ball Tree": "ball_tree", "KD Tree": "kd_tree", "Brute": "brute"}
CLASSIFIER_KNN_WEIGHTS =  {'Uniform': 'uniform', 'Distance': 'distance'}

CLASSIFIER_RF_CRITERION = {"Gini": "gini", "Entropy": "entropy", "Log Loss": "log_loss"}
REGRESSOR_RF_CRITERION = {"Squared Error": "squared_error", "Absolute error": "absolute_error", "Friedman MSE": "friedman_mse", "Poisson": "poisson"}

CLASSIFIER_SCORING = {"Accuracy": "accuracy", "Accuracy Balanced": "balanced_accuracy", "F1 (Binary)": "f1", "F1 Micro": "f1_micro", "F1 Macro": "f1_macro", "F1 Weighted": "f1_weighted", "Precision (Binary)": "precision", "Precision Micro": "precision_micro", "Precision Macro": "precision_macro", "Precision Weighted": "precision_weighted", "Recall (Binary)": "recall", "Recall Micro": "recall_micro", "Recall Macro": "recall_macro", "Recall Weighted": "recall_weighted", "MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error" , "RMSE": "rmse"}
REGRESSOR_SCORING ={"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error" , "RMSE": "rmse", "R2": "r2", "Explained Variance": "explained_variance"}

def apply_classifier_prediction(df, target, train_size, model, params, ts_cross_val=False):
    # Extract the target values
    y = df[target].values

    # Extract the feature values (excluding the target column)
    X = df.drop(columns=[target]).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42, shuffle=not ts_cross_val)
    
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
        
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
        
    return y_train, y_train_pred, y_test, y_test_pred

def apply_classifier(df, target, train_size, model, params, ts_cross_val=False, scoring='f1_macro'):
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
        scores = time_series_cross_val(X, y, cv, clf, scoring=scoring)
    else: 
        scores = cross_val(X, y, cv, clf, scoring=scoring)
        
    if scores.isna().any().any():
        if df[target].nunique() != 2 and (scoring == 'f1' or  scoring == 'precision' or  scoring == 'recall'):
            raise ValueError("Target is multiclass but scoring is 'Binary'. Please choose another scoring.")
        else:   
            raise ValueError("Target contains invalid values. Please check your target.")
    
    return scores

def apply_regressor_prediction(df, target, train_size, model, params, ts_cross_val=False):
    # Extract the target values
    y = df[target].values

    # Extract the feature values (excluding the target column)
    X = df.drop(columns=[target]).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42, shuffle=not ts_cross_val)
    
    # initialize classifier
    if model == REGRESSOR[0]:
        reg = DummyRegressor(**params)
    elif model == REGRESSOR[1]:
        reg = LinearRegression(**params)
    elif model == REGRESSOR[2]:
        reg = RandomForestRegressor(**params)        
    elif model == REGRESSOR[3]:
        reg = XGBRegressor(**params)
        
    reg.fit(X_train, y_train)
    
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    
        
    return y_train, y_train_pred, y_test, y_test_pred
    
    
def apply_regressor(df, target, train_size, model, params, ts_cross_val=False, scoring='neg_mean_squared_error'):
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
        scores = time_series_cross_val(X, y, cv, reg, scoring=scoring)
    else: 
        scores = cross_val(X, y, cv, reg, scoring=scoring)
        
    if scores.isna().any().any():
        raise ValueError("Target contains invalid values. Please check your target.")
    
    return scores
                         
def cross_val(X, y, cv, model, scoring):

    # Perform cross-validation and collect scores
    if scoring == 'rmse':
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    else:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Convert negative mean squared errors to positive 
    if scoring == 'neg_mean_squared_error' or scoring == 'neg_mean_absolute_error':
        scores = -scores
    elif scoring == 'rmse':
        scores = [(-score) ** 0.5 for score in scores]

    # Create a DataFrame with fold indices and scores
    df_scores = pd.DataFrame({'Fold': range(1, len(scores) + 1), 'Score': scores})

    return df_scores

def time_series_cross_val(X, y, cv, model, scoring):
    # Create TimeSeriesSplit cross-validation generator
    tscv = TimeSeriesSplit(n_splits=cv)

    # Perform cross-validation and collect scores
    if scoring == 'rmse':
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    else:
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

    # Convert negative mean squared errors to positive 
    if scoring == 'neg_mean_squared_error' or scoring == 'neg_mean_absolute_error':
        scores = -scores
    elif scoring == 'rmse':
        scores = [(-score) ** 0.5 for score in scores]

    # Create a DataFrame with fold indices and scores
    df_scores = pd.DataFrame({'Fold': range(1, len(scores) + 1), 'Score': scores})

    return df_scores
    
        