# Import necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, ctx
import plotly.express as px
import numpy as np

# import analyse methods
from methods.data_exploration.analyse import *

# import supervised methods
from methods.data_analysis.supervised_learning import *

# import plots
from view.page_helper_components.plots import *

# import app
from view.app import app

# import data
from data import table_data

RESULTS = pd.DataFrame(columns=['Model','Score'])
CURR_RESULT = None
CLASSFIER_COUNT = {"Baseline": 0, "KNN": 0, "Random Forest": 0, "Gradient Boosting": 0}

# update baseline style
@app.callback(
    Output("container_classification_baseline_strategy", "style"),
    Input("dropdown_classification_model", "value"),
    State("container_classification_baseline_strategy", "style")
)
def update_style_baseline(method, style):
    if style is None:
        style = {}
    if method == CLASSFIER[0]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_classification_baseline_constant", "style"),
    Input("dropdown_classification_model", "value"),
    Input("dropdown_classification_baseline_strategy", "value"),
    State("container_classification_baseline_constant", "style")
)
def update_style_baseline_constant(method, strategy, style):
    if style is None:
        style = {}
    if method == CLASSFIER[0] and strategy == list(CLASSIFIER_BASELINE_STRATEGY.keys())[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update knn style
@app.callback(
    Output("container_classification_knn", "style"),
    Input("dropdown_classification_model", "value"),
    State("container_classification_knn", "style")
)
def update_style_knn(method, style):
    if style is None:
        style = {}
    if method == CLASSFIER[1]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update random forest style
@app.callback(
    Output("container_classification_random_forest", "style"),
    Input("dropdown_classification_model", "value"),
    State("container_classification_random_forest", "style")
)
def update_style_rf(method, style):
    if style is None:
        style = {}
    if method == CLASSFIER[2]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style


# update xgboost style
@app.callback(
    Output("container_classification_xgboost", "style"),
    Input("dropdown_classification_model", "value"),
    State("container_classification_xgboost", "style")
)
def update_style_xgboost(method, style):
    if style is None:
        style = {}
    if method == CLASSFIER[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update button styles
@app.callback(
    Output("button_classification_apply", "style"),
    Output("button_classification_show", "style"),
    Input("button_classification_show", "n_clicks"),
    Input("button_classification_apply", "n_clicks"),
    # general
    Input("dropdown_classification_dataset", "value"),
    Input("dropdown_classification_target", "value"),
    Input("slider_classification_train_test_split", "value"),
    Input("dropdown_classification_model", "value"),
    Input("checklist_classification_time_series_crossvalidation", "value"),
    # baseline
    Input("dropdown_classification_baseline_strategy", "value"),
    Input("input_classification_baseline_constant", "value"),
    # knn
    Input("slider_classification_knn_n_neighbors", "value"),
    Input("dropdown_classification_knn_algorithm", "value"),
    Input("dropdown_classification_knn_weights", "value"),
    # random forest
    Input("slider_classification_random_forest_n_estimators", "value"),
    Input("slider_classification_random_forest_criterion", "value"),
    Input("slider_classification_random_forest_max_depth", "value"),
    Input("slider_classification_random_forest_warm_start", "value"),
    # xgboost
    Input("slider_classification_xgboost_n_estimators", "value"),
    Input("slider_classification_xgboost_max_depth", "value"),
    Input("slider_classification_xgboost_learning_rate", "value"),
    # alerts
    State("button_classification_apply", "style"),
    State("button_classification_show", "style"),
    Input("alert_classification_invalid_splits", "is_open"),
    Input("alert_classification_missing_classes", "is_open"),
    Input("alert_classification_invalid_neighbors", "is_open"),
    Input("alert_classification", "is_open"),
)
def update_style_buttons(n_clicks1, n_clicks2, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, style_apply, style_show, is_open_invalid_splits, is_open_missing_classes, is_open_invalid_neighbors, is_open_alert):
    triggered_id = ctx.triggered_id
    if style_apply is None:
        style_apply = {}
    if style_show is None:
        style_show = {}   
           
    if n_clicks1 is None or n_clicks1 == 0 or is_open_invalid_splits or is_open_missing_classes or is_open_invalid_neighbors or is_open_alert:
        style_apply['display'] = 'none'
        style_show['display'] = 'block' 
    elif triggered_id is None or triggered_id == 'button_classification_show':
        style_apply['display'] = 'block'
        style_show['display'] = 'none'   
    else:
        style_show['display'] = 'block'
        style_apply['display'] = 'none'  
    return style_apply, style_show

# apply classifier
@app.callback(
    Output("alert_classification_invalid_splits", "is_open"),
    Output("alert_classification_missing_classes", "is_open"),
    Output("alert_classification_invalid_neighbors", "is_open"),
    Output("alert_classification", "children"),
    Output("alert_classification", "is_open"),
    Output("loading_classification_preview", "children"),
    Input("button_classification_show", "n_clicks"),
    # general
    State("dropdown_classification_dataset", "value"),
    State("dropdown_classification_target", "value"),
    State("slider_classification_train_test_split", "value"),
    State("dropdown_classification_model", "value"),
    State("checklist_classification_time_series_crossvalidation", "value"),
    # baseline
    State("dropdown_classification_baseline_strategy", "value"),
    State("input_classification_baseline_constant", "value"),
    # knn
    State("slider_classification_knn_n_neighbors", "value"),
    State("dropdown_classification_knn_algorithm", "value"),
    State("dropdown_classification_knn_weights", "value"),
    # random forest
    State("slider_classification_random_forest_n_estimators", "value"),
    State("slider_classification_random_forest_criterion", "value"),
    State("slider_classification_random_forest_max_depth", "value"),
    State("slider_classification_random_forest_warm_start", "value"),
    # xgboost
    State("slider_classification_xgboost_n_estimators", "value"),
    State("slider_classification_xgboost_max_depth", "value"),
    State("slider_classification_xgboost_learning_rate", "value"),
)
def update_current_results(n_clicks, dataset_name, target, train_test_split, model, ts_cross_val, baseline_strategy, baseline_constant, knn_n_neighbors, knn_algorithm, knn_weights, rf_n_estimators, rf_criterion, rf_max_depth, rf_warm_start, xgb_n_estimators, xgb_max_depth, xgb_learning_rate):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    # read out parameter
    params = {}
    if model == CLASSFIER[0]: # baseline
        params['strategy'] = CLASSIFIER_BASELINE_STRATEGY[baseline_strategy]
        if baseline_strategy == list(CLASSIFIER_BASELINE_STRATEGY.keys())[3]:
            params['constant'] = baseline_constant
    elif model == CLASSFIER[1]: # knn
        params['n_neighbors'] = knn_n_neighbors
        params['algorithm'] = CLASSIFIER_KNN_ALGORITHM[knn_algorithm]
        params['weights'] = CLASSIFIER_KNN_WEIGHTS[knn_weights]
    elif model == CLASSFIER[2]: # random forest
        params['n_estimators'] = rf_n_estimators
        params['criterion'] = CLASSIFIER_RF_CRITERION[rf_criterion]
        params['warm_start'] = rf_warm_start
        if rf_max_depth == 36:
            params['max_depth'] = None
        else:
            params['max_depth'] = rf_max_depth
    elif model == CLASSFIER[3]: # xgboost
        params['n_estimators'] = xgb_n_estimators
        params['learning_rate'] = xgb_learning_rate
        if xgb_max_depth == 36:
            params['max_depth'] = None
        else:
            params['max_depth'] = xgb_max_depth
           
    df = table_data.ALL_DATASETS[dataset_name]
    
    # use data between defined ranges
    min_index = table_data.ALL_RANGES[dataset_name][0]
    max_index = table_data.ALL_RANGES[dataset_name][1]
    df = df.loc[min_index:max_index].copy()
    
    try:
        scores = apply_classifier(df, target, train_test_split, model, params, ts_cross_val=ts_cross_val)
    except ValueError as e:
        print(e)
        alert_splits = False
        alert_missing_classes = False
        alert_neighbors = False
        alert_str = str(e)
        alert = False
        if alert_str.startswith('n_splits='):
            alert_splits = True
        elif alert_str.startswith('Expected n_neighbors'):
            alert_neighbors = True
        elif "fits failed with the following error" in alert_str:
            alert_missing_classes = True
        else:
            alert = True
            
        return alert_splits, alert_missing_classes, alert_neighbors, alert_str, alert, dash.no_update
    
    figure = get_cross_validation_plot(scores)
    graph = dcc.Graph(id="figure_classification_result", className='graph_categorical', figure=figure)
    
    global CURR_RESULT
    CURR_RESULT = (model, scores["Score"].mean())
 
    return False, False, False, dash.no_update, False, graph

# update summary
@app.callback(
    Output("analysis_classification_summary", "figure"),
    Input("button_classification_apply", "n_clicks"),
)
def update_classification_summary(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    # get current result
    global CURR_RESULT, RESULTS
    curr_model = CURR_RESULT[0]
    curr_score = CURR_RESULT[1]
    
    # increase count
    CLASSFIER_COUNT[curr_model] = CLASSFIER_COUNT[curr_model] + 1
    
    # add index to current model
    curr_model = curr_model + " " + str(CLASSFIER_COUNT[curr_model])
    
    # update all results
    row = pd.DataFrame({'Model': curr_model, 'Score': curr_score}, index=[0])
    RESULTS = pd.concat([row,RESULTS.loc[:]]).reset_index(drop=True)
    print(RESULTS)
    
    # update figure
    figure = get_summary_plot(RESULTS)

    return figure

# update after selected dataset changes
@app.callback(
    Output("dropdown_classification_target", "options"),
    Output("dropdown_classification_target", "value"),
    Input("dropdown_classification_dataset", "value"),
)
def update_after_dataset_changes(dataset_name):
    if dataset_name is None or dataset_name == "":
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    options = list(df.columns)
    value = options[0]
    
    return options, value


