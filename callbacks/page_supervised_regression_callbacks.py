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
REGRESSOR_COUNT = {"Baseline": 1, "Linear": 1, "Random Forest": 1, "Gradient Boosting": 1}

# update baseline style
@app.callback(
    Output("container_regression_baseline_strategy", "style"),
    Input("dropdown_regression_model", "value"),
    State("container_regression_baseline_strategy", "style")
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
    Output("container_regression_baseline_constant", "style"),
    Input("dropdown_regression_model", "value"),
    Input("dropdown_regression_baseline_strategy", "value"),
    State("container_regression_baseline_constant", "style")
)
def update_style_baseline_constant(method, strategy, style):
    if style is None:
        style = {}
    if method == CLASSFIER[0] and strategy == list(REGRESSOR_BASELINE_STRATEGY.keys())[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update random forest style
@app.callback(
    Output("container_regression_random_forest", "style"),
    Input("dropdown_regression_model", "value"),
    State("container_regression_random_forest", "style")
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
    Output("container_regression_xgboost", "style"),
    Input("dropdown_regression_model", "value"),
    State("container_regression_xgboost", "style")
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
    Output("button_regression_apply", "style"),
    Output("button_regression_show", "style"),
    Input("button_regression_show", "n_clicks"),
    Input("button_regression_apply", "n_clicks"),
    # general
    Input("dropdown_regression_dataset", "value"),
    Input("dropdown_regression_target", "value"),
    Input("slider_regression_train_test_split", "value"),
    Input("dropdown_regression_model", "value"),
    Input("checklist_regression_time_series_crossvalidation", "value"),
    Input("dropdown_regression_scoring", "value"),
    # baseline
    Input("dropdown_regression_baseline_strategy", "value"),
    Input("input_regression_baseline_constant", "value"),
    # random forest
    Input("slider_regression_random_forest_n_estimators", "value"),
    Input("slider_regression_random_forest_criterion", "value"),
    Input("slider_regression_random_forest_max_depth", "value"),
    Input("slider_regression_random_forest_warm_start", "value"),
    # xgboost
    Input("slider_regression_xgboost_n_estimators", "value"),
    Input("slider_regression_xgboost_max_depth", "value"),
    Input("slider_regression_xgboost_learning_rate", "value"),
    # alerts
    Input("alert_classification", "is_open"),
    State("button_regression_apply", "style"),
    State("button_regression_show", "style")
)
def update_style_buttons(n_clicks1, n_clicks2, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, is_open_alert, style_apply, style_show):
    triggered_id = ctx.triggered_id
    if style_apply is None:
        style_apply = {}
    if style_show is None:
        style_show = {}   
           
    if n_clicks1 is None or n_clicks1 == 0 or is_open_alert:
        style_apply['display'] = 'none'
        style_show['display'] = 'block' 
    elif triggered_id is None or triggered_id == 'button_regression_show':
        style_apply['display'] = 'block'
        style_show['display'] = 'none'   
    else:
        style_show['display'] = 'block'
        style_apply['display'] = 'none'  
    return style_apply, style_show

# apply regressor
@app.callback(
    Output("alert_regression", "children"),
    Output("alert_regression", "is_open"),
    Output("loading_regression_preview", "children"),
    Input("button_regression_show", "n_clicks"),
    # general
    State("dropdown_regression_dataset", "value"),
    State("dropdown_regression_target", "value"),
    State("slider_regression_train_test_split", "value"),
    State("dropdown_regression_model", "value"),
    State("checklist_regression_time_series_crossvalidation", "value"),
    State("dropdown_regression_scoring", "value"),
    # baseline
    State("dropdown_regression_baseline_strategy", "value"),
    State("input_regression_baseline_constant", "value"),
    # random forest
    State("slider_regression_random_forest_n_estimators", "value"),
    State("slider_regression_random_forest_criterion", "value"),
    State("slider_regression_random_forest_max_depth", "value"),
    State("slider_regression_random_forest_warm_start", "value"),
    # xgboost
    State("slider_regression_xgboost_n_estimators", "value"),
    State("slider_regression_xgboost_max_depth", "value"),
    State("slider_regression_xgboost_learning_rate", "value"),
)
def update_current_results(n_clicks, dataset_name, target, train_test_split, model, ts_cross_val, scoring, baseline_strategy, baseline_constant, rf_n_estimators, rf_criterion, rf_max_depth, rf_warm_start, xgb_n_estimators, xgb_max_depth, xgb_learning_rate):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    # read out parameter
    params = {}
    if model == REGRESSOR[0]: # baseline
        params['strategy'] = REGRESSOR_BASELINE_STRATEGY[baseline_strategy]
        if baseline_strategy == list(REGRESSOR_BASELINE_STRATEGY.keys())[3]:
            params['constant'] = baseline_constant
    elif model == REGRESSOR[1]: # linear
        pass
    elif model == REGRESSOR[2]: # random forest
        params['n_estimators'] = rf_n_estimators
        params['criterion'] = REGRESSOR_RF_CRITERION[rf_criterion]
        params['warm_start'] = rf_warm_start
        if rf_max_depth == 36:
            params['max_depth'] = None
        else:
            params['max_depth'] = rf_max_depth
    elif model == REGRESSOR[3]: # xgboost
        params['n_estimators'] = xgb_n_estimators
        params['learning_rate'] = xgb_learning_rate
        if xgb_max_depth == 36:
            params['max_depth'] = None
        else:
            params['max_depth'] = xgb_max_depth
           
    df = table_data.ALL_DATASETS[dataset_name]
    
    scoring = REGRESSOR_SCORING[scoring]
    
    # use data between defined ranges
    min_index = table_data.ALL_RANGES[dataset_name][0]
    max_index = table_data.ALL_RANGES[dataset_name][1]
    df = df.loc[min_index:max_index].copy()
    
    if len(ts_cross_val) == 0:
        ts_cross_val = False
    else:
        ts_cross_val = True
    
    try:
        scores = apply_regressor(df, target, train_test_split, model, params, ts_cross_val=ts_cross_val, scoring=scoring)
    except ValueError as e:
        print(e)
        alert = True
        alert_str = str(e)
        return alert_str, alert, dash.no_update
    
    figure = get_cross_validation_plot(scores, title="Results Cross Validation Scores")
    graph = dcc.Graph(id="figure_regression_result", className='graph_categorical', figure=figure)
    
    global CURR_RESULT
    CURR_RESULT = (model, scores["Score"].mean())
 
    return dash.no_update, False, graph

# update summary
@app.callback(
    Output("analysis_regression_summary", "figure"),
    Output("input_regression_model_name", "value"),
    Input("button_regression_apply", "n_clicks"),
    State("input_regression_model_name", "value"),
    State("dropdown_regression_model", "value"),
    State("dropdown_regression_scoring", "value"),
)
def update_regression_summary(n_clicks, model_name, model, scoring):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    # get current result
    global CURR_RESULT, RESULTS
    curr_score = CURR_RESULT[1]
    
    # update all results
    row = pd.DataFrame({'Model': model_name, 'Score': curr_score}, index=[0])
    RESULTS = pd.concat([row,RESULTS.loc[:]]).reset_index(drop=True)
    
    # update figure
    figure = get_summary_plot(RESULTS)

    REGRESSOR_COUNT[model] += 1 
    
    model_name = model + " " + str(REGRESSOR_COUNT[model]) + " " + scoring

    return figure, model_name

# update after selected dataset changes
@app.callback(
    Output("dropdown_regression_target", "options"),
    Output("dropdown_regression_target", "value"),
    Input("dropdown_regression_dataset", "value"),
)
def update_after_dataset_changes(dataset_name):
    if dataset_name is None or dataset_name == "":
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    options = list(df.columns)
    value = options[0]
    
    return options, value


