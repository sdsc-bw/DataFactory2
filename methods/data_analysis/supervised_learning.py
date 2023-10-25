import numpy as np
import pandas as pd
#from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error, r2_score, explained_variance_score, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle



CLASSFIER = ["Baseline", "KNN", "Random Forest"]
REGRESSOR = ["Baseline", "Linear", "Random Forest"]

CLASSIFIER_BASELINE_STRATEGY = {"Prior": "prior", "Most Frequent": "most_frequent", "Stratified": "stratified", "Constant": "constant", "Look Back": "look_back"}
REGRESSOR_BASELINE_STRATEGY = {"Mean": "mean", "Median": "median", "Quantile": "quantile", "Constant": "constant", "Look Back": "look_back"}

CLASSIFIER_KNN_ALGORITHM = {"Auto": "auto", "Ball Tree": "ball_tree", "KD Tree": "kd_tree", "Brute": "brute"}
CLASSIFIER_KNN_WEIGHTS =  {'Uniform': 'uniform', 'Distance': 'distance'}

CLASSIFIER_RF_CRITERION = {"Gini": "gini", "Entropy": "entropy", "Log Loss": "log_loss"}
REGRESSOR_RF_CRITERION = {"MSE": "mse", "Friedman MSE": "friedman_mse", "Poisson": "poisson"}

CLASSIFIER_SCORING = {"Accuracy": "accuracy", "Accuracy Balanced": "balanced_accuracy", "F1 (Binary)": "f1", "F1 Micro": "f1_micro", "F1 Macro": "f1_macro", "F1 Weighted": "f1_weighted", "Precision (Binary)": "precision", "Precision Micro": "precision_micro", "Precision Macro": "precision_macro", "Precision Weighted": "precision_weighted", "Recall (Binary)": "recall", "Recall Micro": "recall_micro", "Recall Macro": "recall_macro", "Recall Weighted": "recall_weighted", "MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error" , "RMSE": "rmse"}
REGRESSOR_SCORING ={"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error" , "R2": "r2", "Explained Variance": "explained_variance"}

CLASSIFIER_LINKS = ['https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html', '', 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html']
CLASSIFIER_DESCRIPTION = ['Read more', 'This method allows to use the mean of the previous values as baseline.', 'Read more', 'Read more']

REGRESSOR_LINKS = ['https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html', '', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html']
REGRESSOR_DESCRIPTION = ['Read more', 'This method allows to use the mean of the previous values as baseline.', 'Read more', 'Read more']

TS_CROSS_VALIDATION_LINKS = ['https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html']
TS_CROSS_VALIDATION_DESCRIPTION = ['Read more']

def apply_classifier_prediction(df, target, train_size, model, params, ts_cross_val=False):
    # Extract the target values
    y = df[target].values

    # Extract the feature values (excluding the target column)
    X = df.drop(columns=[target]).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42, shuffle=not ts_cross_val)
    
    # initialize classifier
    if 'look_back' in list(params.keys()):
        look_back = params['look_back']
        y_train_pred = compute_median_previous_n(y_train, look_back)
        y_test_pred = compute_median_previous_n(y_test, look_back)
        
        assert len(y_train_pred) == len(y_train)
        assert len(y_test_pred) == len(y_test)
        
    else:
        if model == CLASSFIER[0]:
            clf = DummyClassifier(**params)
        elif model == CLASSFIER[1]:
            clf = KNeighborsClassifier(**params)
        elif model == CLASSFIER[2]:
            clf = RandomForestClassifier(**params)        
        #elif model == CLASSFIER[3]:
        #    clf = XGBClassifier(**params)
        
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
    # extract feature importance
    feature_importance = []
    feature_names = list(df.columns)
    
    if isinstance(clf, RandomForestClassifier):
        if hasattr(clf, "feature_importances_"):
            for feature_name, importance_score in zip(feature_names, clf.feature_importances_):
                dict_importance = {}
                dict_importance['Feature'] = feature_name
                dict_importance['Importance'] = importance_score
                feature_importance.append(dict_importance)
    elif model == CLASSFIER[0]:
        for feature_name in feature_names:
            dict_importance = {}
            dict_importance['Feature'] = feature_name
            if target == feature_name:
                dict_importance['Importance'] = 1.0
            else:
                dict_importance['Importance'] = 0.0
            feature_importance.append(dict_importance)
    
        
    return y_train, y_train_pred, y_test, y_test_pred, feature_importance

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
    if 'look_back' in list(params.keys()):
        look_back = params['look_back']
        y_pred = compute_median_previous_n(y, look_back)

        assert len(y) == len(y_pred)
        
        # simulate cross validation
        scores = []
        kf = KFold(n_splits=cv)
        for train_indices, test_indices in kf.split(X):
            y_split = y[test_indices]
            y_pred_split = y_pred[test_indices]
            scores.append(score_prediction(y_split, y_pred_split, scoring))
            
        scores = pd.DataFrame({'Fold': range(1, len(scores) + 1), 'Score': scores})

    else:
        if model == CLASSFIER[0]:
            clf = DummyClassifier(**params)
        elif model == CLASSFIER[1]:
            clf = KNeighborsClassifier(**params)
        elif model == CLASSFIER[2]:
            clf = RandomForestClassifier(**params)        
        #elif model == CLASSFIER[3]:
        #    clf = XGBClassifier(**params)

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
    
    # initialize regressor
    if 'look_back' in list(params.keys()):
        look_back = params['look_back']
        y_train_pred = compute_mean_previous_n(y_train, look_back)
        y_test_pred = compute_mean_previous_n(y_test, look_back)
        
        assert len(y_train_pred) == len(y_train)
        assert len(y_test_pred) == len(y_test)
        
    else:
        if model == REGRESSOR[0]:
            reg = DummyRegressor(**params)
        elif model == REGRESSOR[1]:
            reg = LinearRegression(**params)
        elif model == REGRESSOR[2]:
            reg = RandomForestRegressor(**params)        
        #elif model == REGRESSOR[3]:
        #    reg = XGBRegressor(**params)

        reg.fit(X_train, y_train)

        y_train_pred = reg.predict(X_train)
        y_test_pred = reg.predict(X_test)
        
    # extract feature importance
    feature_importance = []
    feature_names = list(df.columns)

    if isinstance(reg, RandomForestRegressor):
        if hasattr(reg, "feature_importances_"):
            for feature_name, importance_score in zip(feature_names, reg.feature_importances_):
                dict_importance = {}
                dict_importance['Feature'] = feature_name
                dict_importance['Importance'] = importance_score
                feature_importance.append(dict_importance)
    elif isinstance(reg, LinearRegression):
        if hasattr(reg, "coef_"):
            for feature_name, coef in zip(feature_names, reg.coef_):
                dict_importance = {}
                dict_importance['Feature'] = feature_name
                dict_importance['Importance'] = abs(coef)
                feature_importance.append(dict_importance)
    elif model == REGRESSOR[0]:
        for feature_name in feature_names:
            dict_importance = {}
            dict_importance['Feature'] = feature_name
            if target == feature_name:
                dict_importance['Importance'] = 1.0
            else:
                dict_importance['Importance'] = 0.0
            feature_importance.append(dict_importance)
    
        
    return y_train, y_train_pred, y_test, y_test_pred, feature_importance
    
    
def apply_regressor(df, target, train_size, model, params, ts_cross_val=False, scoring='neg_mean_squared_error'):
    # Extract the target values
    y = df[target].values

    # Extract the feature values (excluding the target column)
    X = df.drop(columns=[target]).values
    
    # compute number of folds
    cv = int(1.0 / (1 - train_size))
    if cv < 2:
        cv = 2
    
    # initialize regressor
    if 'look_back' in list(params.keys()):
        look_back = params['look_back']
        y_pred = compute_mean_previous_n(y, look_back)

        assert len(y) == len(y_pred)
        
        # simulate cross validation
        scores = []
        kf = KFold(n_splits=cv)
        for train_indices, test_indices in kf.split(X):
            y_split = y[test_indices]
            y_pred_split = y_pred[test_indices]
            scores.append(score_prediction(y_split, y_pred_split, scoring))
            
        scores = pd.DataFrame({'Fold': range(1, len(scores) + 1), 'Score': scores})

    else:
        if model == REGRESSOR[0]:
            reg = DummyRegressor(**params)
        elif model == REGRESSOR[1]:
            reg = LinearRegression(**params)
        elif model == REGRESSOR[2]:
            reg = RandomForestRegressor(**params)        
        #elif model == REGRESSOR[3]:
        #    reg = XGBRegressor(**params)

        # evalutate classifier
        if ts_cross_val:
            scores = time_series_cross_val(X, y, cv, reg, scoring=scoring)
        else: 
            scores = cross_val(X, y, cv, reg, scoring=scoring)

        if scores.isna().any().any():
            raise ValueError("Target contains invalid values. Please check your target.")
    
    return scores
                         
def cross_val(X, y, cv, model, scoring):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scoring == 'rmse':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_score = mean_squared_error(y_test, y_pred, squared=False)
        else:
            fold_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()

        scores.append(fold_score)

    # Convert negative mean squared errors to positive 
    if scoring == 'neg_mean_squared_error' or scoring == 'neg_mean_absolute_error':
        scores = [(-score) for score in scores]
    elif scoring == 'rmse':
        scores = [(-score) ** 0.5 for score in scores]

    df_scores = pd.DataFrame({'Fold': range(1, cv + 1), 'Score': scores})

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


def compute_mean_previous_n(lst, n):
    result = []
    total_sum = 0.0

    for i, value in enumerate(lst):
        if n == 1:
            if i == 0:
                result.append(0.0)
            else:
                result.append(lst[i - 1])
        else:
            if i < n:
                total_sum += value
                mean = total_sum / (i + 1)
            else:
                total_sum += value - lst[i - n]
                mean = total_sum / n
            result.append(mean)

    result = np.array(result)    
        
    return result    

def compute_median_previous_n(lst, n):
    result = []

    for i, value in enumerate(lst):
        if n == 1:
            if i == 0:
                result.append(0)  # Assuming input and output values are integers
            else:
                result.append(lst[i - 1])
        else:
            if i < n:
                result.append(np.median(lst[:i + 1]))
            else:
                result.append(np.median(lst[i - n + 1:i + 1]))

    result = np.array(result)
        
    return result
        
def score_prediction(y, y_pred, scoring):
    if scoring == 'accuracy':
        score = accuracy_score(y, y_pred)
    elif scoring == 'balanced_accuracy':
        score = balanced_accuracy_score(y, y_pred)
    elif scoring == 'f1':
        score = f1_score(y, y_pred, average="binary")
    elif scoring == 'f1_micro':
        score = f1_score(y, y_pred, average="micro")
    elif scoring == 'f1_macro':
        score = f1_score(y, y_pred, average="macro")
    elif scoring == 'f1_weighted':
        score = f1_score(y, y_pred, average="weighted")
    elif scoring == 'precision':
        score = precision_score(y, y_pred, average="binary")
    elif scoring == 'precision_micro':
        score = precision_score(y, y_pred, average="micro")
    elif scoring == 'precision_macro':
        score = precision_score(y, y_pred, average="macro")
    elif scoring == 'precision_weighted':
        score = precision_score(y, y_pred, average="weighted")
    elif scoring == 'recall':
        score = recall_score(y, y_pred, average="binary")
    elif scoring == 'recall_micro':
        score = recall_score(y, y_pred, average="micro")
    elif scoring == 'recall_macro':
        score = recall_score(y, y_pred, average="macro")
    elif scoring == 'recall_weighted':
        score = recall_score(y, y_pred, average="weighted")
    elif scoring == 'neg_mean_absolute_error':
        score = mean_absolute_error(y, y_pred)
    elif scoring == 'neg_mean_squared_error':
        score = mean_squared_error(y, y_pred)
    elif scoring == 'rmse':
        score = mean_squared_error(y, y_pred, squared=False)
    elif scoring == 'r2':
        score = r2_score(y, y_pred)
    elif scoring == 'explained_variance':
        score = explained_variance_score(y, y_pred)
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")
        
    return score