import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


CLASSFIER = ["Baseline", "KNN", "Random Forest", "Gradient Boosting"]
REGRESSOR = ["Baseline", "Linear", "Random Forest", "Gradient Boosting"]

CLASSIFIER_BASELINE_STRATEGY = {"Prior": "prior", "Most Frequent": "most_frequent", "Stratified": "stratified", "Constant": "constant"}
REGRESSOR_BASELINE_STRATEGY = {"Mean": "mean", "Median": "median", "Quantile": "quantile", "Constant": "constant"}

CLASSIFIER_KNN_ALGORITHM = {"Auto": "auto", "Ball Tree": "ball_tree", "KD Tree": "kd_tree", "Brute": "brute"}
CLASSIFIER_KNN_WEIGHTS =  {'Uniform': 'uniform', 'Distance': 'distance'}

CLASSIFIER_RF_CRITERION = {"Gini": "gini", "Entropy": "entropy", "Log Loss": "log_loss"}
REGRESSOR_RF_CRITERION = {"Squared Error": "squared_error", "Absolute error": "absolute_error", "Friedman MSE": "friedman_mse", "Poisson": "poisson"}

def apply_classifier(df, target, train_size, model, params):
    X = df.copy(deep=True)
    y = X.pop(target) 
    
    # compute number of folds
    cv = int(1.0 / (1 - train_size))
    
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
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    
    folds = []
    for i in range(len(scores)):
        folds.append(f"Fold {i+1}")
    df_scores = pd.DataFrame({'Fold': folds, 'Score': scores})
    
    return df_scores

def apply_regressor(df, target, train_size, model, params):
    X = df.copy(deep=True)
    y = X.pop(target) 
    
    # compute number of folds
    cv = int(1.0 / (1 - train_size))
    
    # initialize classifier
    if model == REGRESSOR[0]:
        clf = DummyRegressor(**params)
    elif model == REGRESSOR[1]:
        clf = LinearRegression(**params)
    elif model == REGRESSOR[2]:
        clf = RandomForestRegressor(**params)        
    elif model == REGRESSOR[3]:
        clf = XGBRegressor(**params)
    else:
        print(f"Unknown model: {model}")
        
    # evalutate classifier
    scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error')
    scores = -scores
    
    folds = []
    for i in range(len(scores)):
        folds.append(f"Fold {i+1}")
    df_scores = pd.DataFrame({'Fold': folds, 'Score': scores})
    
    return df_scores
    
        