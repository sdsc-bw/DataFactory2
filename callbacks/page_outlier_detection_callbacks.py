# Import necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, ctx
import plotly.express as px
import numpy as np

# import plots
from view.page_helper_components.plots import *

# import app
from view.app import app

# import data
from data import table_data

# import detection methods
from methods.data_exploration.outlier_detection import *

# import analyse methods
from methods.data_exploration.analyse import *

# import plots
from view.page_helper_components.plots import *


# update random forest detector style
@app.callback(
    Output("container_outlier_random_forest", "style"),
    Input("dropdown_outlier_method", "value"),
    State("container_outlier_random_forest", "style")
)
def update_style_rf(method, style):
    if style is None:
        style = {}
    if method == OUTLIER_DETECTION_METHODS[0]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update density detector style
@app.callback(
    Output("container_outlier_densitiy", "style"),
    Input("dropdown_outlier_method", "value"),
    State("container_outlier_densitiy", "style")
)
def update_style_density(method, style):
    if style is None:
        style = {}
    if method == OUTLIER_DETECTION_METHODS[1]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update kv detector style
@app.callback(
    Output("container_outlier_kv", "style"),
    Input("dropdown_outlier_method", "value"),
    State("container_outlier_kv", "style")
)
def update_style_kv(method, style):
    if style is None:
        style = {}
    if method == OUTLIER_DETECTION_METHODS[2]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update button styles
@app.callback(
    Output("button_outlier_apply", "style"),
    Output("button_outlier_show", "style"),
    Input("button_outlier_show", "n_clicks"),
    Input("button_outlier_apply", "n_clicks"),
    Input("dropdown_outlier_method", "value"),
    Input("slider_outlier_random_forest_n_estimators", "value"),
    Input("slider_outlier_densitiy_n_neighbors", "value"),
    Input("dropdown_outlier_densitiy_algorithm", "value"),
    Input("dropdown_outlier_kv_feature", "value"),
    State("button_outlier_apply", "style"),
    State("button_outlier_show", "style")
)
def update_style_buttons(n_clicks1, n_clicks2, v1, v2, v3, v4, v5, style_apply, style_show):
    triggered_id = ctx.triggered_id
    if style_apply is None:
        style_apply = {}
    if style_show is None:
        style_show = {}   
        
    if n_clicks1 is None or n_clicks1 == 0:
        style_apply['display'] = 'none'
        style_show['display'] = 'block' 
    elif triggered_id is None or triggered_id == 'button_outlier_show':
        style_apply['display'] = 'block'
        style_show['display'] = 'none'   
    else:
        style_show['display'] = 'block'
        style_apply['display'] = 'none'  
    return style_apply, style_show

# update outlier plot
@app.callback(
    Output("loading_outlier_preview", "children"),
    Output("table_outlier_detection", "data"),
    Output("table_outlier_detection", "columns"),
    Output("table_outlier_detection", "selected_rows"),
    Input("button_outlier_show", "n_clicks"),
    State("dropdown_outlier_method", "value"),
    State("slider_outlier_random_forest_n_estimators", "value"),
    State("slider_outlier_densitiy_n_neighbors", "value"),
    State("dropdown_outlier_densitiy_algorithm", "value"),
    State("dropdown_outlier_kv_feature", "value"),
    State("table_outlier_detection", "data"),
)
def update_outlier_plot(n_clicks, method, rf_n_estimators, densitiy_n_neighbors, densitiy_algorithm, kv_feature, data):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    # read out parameter
    params = {}
    if method == OUTLIER_DETECTION_METHODS[0]: # isolation forest detector
        params['n_estimators'] = rf_n_estimators
    elif method == OUTLIER_DETECTION_METHODS[1]: # density detector
        params['n_neighbors'] = densitiy_n_neighbors
        params['algorithm'] = OUTLIER_DETECTION_LOCAL_ALGORITHM[densitiy_algorithm]
    elif method == OUTLIER_DETECTION_METHODS[2]: # kv detector
        params['feature'] = kv_feature
    
    # apply detector       
    df_num = table_data.DF_RAW.select_dtypes(include=NUMERICS)
    df_outlier, is_outlier= apply_outlier_detection(df_num, method, params)
    
    # update figure
    figure = get_outlier_plot(df_outlier)
    graph = dcc.Graph(id="figure_outlier_preview", className='graph_categorical', figure=figure)
    
    # update outlier datatable
    indices = is_outlier.index[is_outlier == True]
    df_outlier = df_num.loc[indices]
    df_outlier = df_outlier.reset_index()    

    data_datatable = [{col: df_outlier.loc[i, col] for col in df_outlier.columns} for i in df_outlier.index]
    columns_datatable = [{"name": col, "id": col} for col in df_outlier.columns]   
    selected_rows = list(range(len(df_outlier)))
    
    return graph, data_datatable, columns_datatable, selected_rows

# update outlier plot
@app.callback(
    Output("table_outlier_detection", "data"),
    Output("figure_outlier_preview", "figure"),
    # update overview page
    Output("datatable_overview", "data"),
    Output("datatable_overview", "columns"),
    # inputes
    Input("button_outlier_apply", "n_clicks"),
    State("dropdown_outlier_method", "value"),
    State("slider_outlier_random_forest_n_estimators", "value"),
    State("slider_outlier_densitiy_n_neighbors", "value"),
    State("dropdown_outlier_densitiy_algorithm", "value"),
    State("dropdown_outlier_kv_feature", "value"),
    State("table_outlier_detection", "data"), 
    State("table_outlier_detection", "selected_rows"),
)
def update_outlier_df(n_clicks, method, rf_n_estimators, densitiy_n_neighbors, densitiy_algorithm, kv_feature, data, selected_rows):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    # read out parameter
    params = {}
    if method == OUTLIER_DETECTION_METHODS[0]: # isolation forest detector
        params['n_estimators'] = rf_n_estimators
    elif method == OUTLIER_DETECTION_METHODS[1]: # density detector
        params['n_neighbors'] = densitiy_n_neighbors
        params['algorithm'] = OUTLIER_DETECTION_LOCAL_ALGORITHM[densitiy_algorithm]
    elif method == OUTLIER_DETECTION_METHODS[2]: # kv detector
        params['feature'] = kv_feature
    
    # update dataframe
    indices = []
    for i, d in enumerate(data):
        if i in selected_rows:
            indices.append(d['index'])
    table_data.DF_RAW = table_data.DF_RAW.drop(indices)
    
    data = []
    figure = {}
    
    # update datatable
    df = analyse_df(table_data.DF_RAW)
    data_datatable_overview = [{col: df.loc[i, col] for col in df.columns} for i in df.index]
    columns_datatable_overview = [{"name": col, "id": col} for col in df.columns]   
    
    return data, figure, data_datatable_overview, columns_datatable_overview
