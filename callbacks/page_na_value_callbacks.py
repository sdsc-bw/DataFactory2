# Import necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, ctx
import plotly.express as px
import numpy as np

# import filling methods
from methods.data_exploration.imputer import *

# import analyse methods
from methods.data_exploration.analyse import *

# import plots
from view.page_helper_components.plots import *

# import sliders
from view.page_helper_components.sliders import *

# import app
from view.app import app

# import data
from data import table_data

# import utility
from methods.util import get_nan_positions


# update simple style
@app.callback(
    Output("container_na_simple", "style", allow_duplicate=True),
    Input("dropdown_na_method", "value"),
    State("container_na_simple", "style"),
    prevent_initial_call=True
)
def update_style_simple(method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[0]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_na_simple_fill_value", "style", allow_duplicate=True),
    Input("dropdown_na_simple_strategy", "value"),
    Input("dropdown_na_method", "value"),
    State("container_na_simple_fill_value", "style"),
    prevent_initial_call=True
)
def update_style_simple_fill_value(strategy, method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[0] and strategy == list(IMPUTER_STRATEGIES.keys())[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_na_imputer", "style", allow_duplicate=True),
    Input("dropdown_na_feature", "options"),
    State("container_na_imputer", "style"),
    prevent_initial_call=True
)
def update_categorical_plot(options, style):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    if style is None:
        style = {} 

    if options == []:
        style['display'] = 'none' 
    
    return style


# update iterative style
@app.callback(
    Output("container_na_iterative", "style", allow_duplicate=True),
    Input("dropdown_na_method", "value"),
    State("container_na_iterative", "style"),
    prevent_initial_call=True
)
def update_style_iterative(method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[1]:
        style['display'] = 'block'
    else:
        style['display'] = 'none' 
    return style

@app.callback(
    Output("container_na_iterative_fill_value", "style", allow_duplicate=True),
    Input("dropdown_na_iterative_initial_strategy", "value"),
    Input("dropdown_na_method", "value"),
    State("container_na_iterative_fill_value", "style"),
    prevent_initial_call=True
)
def update_style_iterative_fill_value(strategy, method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[1] and strategy == list(IMPUTER_STRATEGIES.keys())[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update knn style
@app.callback(
    Output("container_na_knn", "style", allow_duplicate=True),
    Input("dropdown_na_method", "value"),
    State("container_na_knn", "style"),
    prevent_initial_call=True
)
def update_style_knn(method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[2]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update manual style
@app.callback(
    Output("container_na_manual", "style", allow_duplicate=True),
    Input("dropdown_na_method", "value"),
    State("container_na_manual", "style"),
    prevent_initial_call=True
)
def update_style_manual(method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'   
    return style

# update fill value simple imputer
@app.callback(
    Output("input_na_manual_fill_value", "style", allow_duplicate=True),
    Input("dropdown_na_method", "value"),
    State("input_na_manual_fill_value", "style"),
    prevent_initial_call=True
)
def update_style_fill_value(method, style):
    if style is None:
        style = {}
    if method == IMPUTER_METHODS[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'   
    return style

# update button styles
@app.callback(
    Output("button_na_apply", "style", allow_duplicate=True),
    Output("button_na_show", "style", allow_duplicate=True),
    Input("button_na_show", "n_clicks"),
    Input("button_na_apply", "n_clicks"),
    Input("dropdown_na_feature", "value"),
    Input("dropdown_na_method", "value"),
    Input("dropdown_na_simple_strategy", "value"),
    Input("input_na_simple_fill_value", "value"),
    Input("dropdown_na_iterative_filling_feature", "value"),
    Input("slider_na_iterative_max_iter", "value"),
    Input("dropdown_na_iterative_tol", "value"),
    Input("slider_na_iterative_n_nearest_features", "value"),
    Input("dropdown_na_iterative_initial_strategy", "value"),
    Input("dropdown_na_iterative_imputation_order", "value"),
    Input("dropdown_na_knn_filling_feature", "value"),
    Input("slider_na_knn_n_neighbors", "value"),
    Input("dropdown_na_knn_weights", "value"),
    Input("input_na_manual_index", "value"),
    Input("input_na_manual_fill_value", "value"),
    State("button_na_apply", "style"),
    State("button_na_show", "style"),
    prevent_initial_call=True
)
def update_style_buttons(n_clicks1, n_clicks2, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, style_apply, style_show):
    triggered_id = ctx.triggered_id
    if style_apply is None:
        style_apply = {}
    if style_show is None:
        style_show = {}   
        
    if triggered_id is None or triggered_id == 'button_na_show':
        style_apply['display'] = 'block'
        style_show['display'] = 'none'   
    else:
        style_show['display'] = 'block'
        style_apply['display'] = 'none'  
    return style_apply, style_show

# update line plot
@app.callback(
    Output("loading_na_imputer_preview", "children", allow_duplicate=True),
    Input("button_na_show", "n_clicks"),
    State("dropdown_na_feature", "value"),
    State("dropdown_na_method", "value"),
    State("dropdown_na_simple_strategy", "value"),
    State("input_na_simple_fill_value", "value"),    
    State("dropdown_na_iterative_filling_feature", "value"),
    State("slider_na_iterative_max_iter", "value"),
    State("dropdown_na_iterative_tol", "value"),
    State("slider_na_iterative_n_nearest_features", "value"),
    State("dropdown_na_iterative_initial_strategy", "value"),
    State("dropdown_na_iterative_fill_value", "value"),
    State("dropdown_na_iterative_imputation_order", "value"),
    State("dropdown_na_knn_filling_feature", "value"),
    State("slider_na_knn_n_neighbors", "value"),
    State("dropdown_na_knn_weights", "value"),
    State("input_na_manual_index", "value"),
    State("input_na_manual_fill_value", "value"),
    prevent_initial_call=True
)
def update_line_plot_after_imputing(n_clicks, col, method, simple_strategy, simple_fill_value, iterative_filling_feature, iterative_max_iter, iterative_tol, iterative_n_nearest_features, iterative_initial_strategy, iterative_fill_value, iterative_imputation_order, knn_filling_feature, knn_n_neighbors, knn_weights, manual_index, manual_fill_value):
    if col is None:
        return dash.no_update
    
    # read out parameter
    params = {}
    if method == IMPUTER_METHODS[0]: # simple imputer
        params['missing_values'] = np.nan
        params['strategy'] = IMPUTER_STRATEGIES[simple_strategy]
        if simple_strategy == list(IMPUTER_STRATEGIES.keys())[3]:
            params['fill_value'] = simple_fill_value
    elif method == IMPUTER_METHODS[1]: # iterativ imputer
        col = [col] + iterative_filling_feature
        params['missing_values'] = np.nan
        params['max_iter'] = iterative_max_iter
        params['tol'] = iterative_tol
        if iterative_n_nearest_features < len(iterative_filling_feature):
            params['n_nearest_features'] = iterative_n_nearest_features
        else:
            params['n_nearest_features'] = None
        params['initial_strategy'] = IMPUTER_STRATEGIES[iterative_initial_strategy]
        if iterative_initial_strategy == list(IMPUTER_STRATEGIES.keys())[3]:
            params['fill_value'] = iterative_fill_value
        params['imputation_order'] = IMPUTER_ORDER[iterative_imputation_order]
    elif method == IMPUTER_METHODS[2]: # knn imputer
        col = [col] + knn_filling_feature
        params['missing_values'] = np.nan
        params['n_neighbors'] = knn_n_neighbors
        params['weights'] = IMPUTER_WEIGHTS[knn_weights]
    elif method == IMPUTER_METHODS[3]: # manual
        params['index'] = manual_index
        params['fill_value'] = manual_fill_value
    
    # apply imputing       
    df_num = table_data.DF_RAW.select_dtypes(include=NUMERICS)
    df_num = apply_imputing(df_num, col, method, params)
    
    # remove helping columns
    if method == IMPUTER_METHODS[1] or method == IMPUTER_METHODS[2]:
        col = col[0]
    
    # get nan positions in the dataframe
    if method != IMPUTER_METHODS[3]:
        nan_positions = get_nan_positions(table_data.DF_RAW, col)[0]
    else:
        nan_positions = [manual_index]
    
    figure = get_imputer_line_plot(df_num, col, nan_positions)
    graph = dcc.Graph(id="figure_na_imputer_preview", className='graph_categorical', figure=figure)
    
    return graph

# update selected features
@app.callback(
    Output('dropdown_na_knn_filling_feature', 'value', allow_duplicate=True),
    Input("checkbox_na_knn_filling_feature", "value"),
    State('dropdown_na_knn_filling_feature', 'options'),
    prevent_initial_call=True
)
def update_selected_features(all_features, options):
    if all_features is None or all_features == []:
        return dash.no_update
    
    return options

# update selected features
@app.callback(
    Output('dropdown_na_iterative_filling_feature', 'value', allow_duplicate=True),
    Input("checkbox_na_iterative_filling_feature", "value"),
    State('dropdown_na_iterative_filling_feature', 'options'),
    prevent_initial_call=True
)
def update_selected_features(all_features, options):
    if all_features is None or all_features == []:
        return dash.no_update
    
    return options

# update features
@app.callback(
    Output('dropdown_na_iterative_filling_feature', 'value', allow_duplicate=True),
    Output('dropdown_na_iterative_filling_feature', 'options', allow_duplicate=True),
    Input("dropdown_na_feature", "value"),
    Input("dropdown_na_feature", "options"),
    prevent_initial_call=True
)
def update_selected_features(na_value, na_options):
    if na_options is None or na_options == []:
        return dash.no_update
    
    options = list(table_data.DF_RAW.columns)

    options.remove(na_value)
    
    return options, options

# update n nearest features
@app.callback(
    Output('slider_na_iterative_n_nearest_features', 'value', allow_duplicate=True),
    Output('slider_na_iterative_n_nearest_features', 'marks', allow_duplicate=True),
    Input("dropdown_na_iterative_filling_feature", "value"),
    prevent_initial_call=True
)
def update_selected_features(filling_features):
    if filling_features is None or filling_features == []:
        return dash.no_update
    
    value = len(filling_features)
    max_features = len(filling_features) - 1
    marks = get_slider_marks_nearest_feature(max_features)
    
    return value, marks

# update features
@app.callback(
    Output('dropdown_na_knn_filling_feature', 'value', allow_duplicate=True),
    Output('dropdown_na_knn_filling_feature', 'options', allow_duplicate=True),
    Input("dropdown_na_feature", "value"),
    Input("dropdown_na_feature", "options"),
    prevent_initial_call=True
)
def update_selected_features(na_value, na_options):
    if na_options is None or na_options == []:
        return dash.no_update
    
    options = list(table_data.DF_RAW.columns)
    
    options.remove(na_value)
    
    return options, options

# update line plot
@app.callback(
    # update parameter
    Output("dropdown_na_feature", "options", allow_duplicate=True),
    Output("dropdown_na_feature", "value", allow_duplicate=True),
    Output("slider_na_iterative_n_nearest_features", "max", allow_duplicate=True),
    Output("slider_na_iterative_n_nearest_features", "value", allow_duplicate=True),
    Output("slider_na_iterative_n_nearest_features", "marks", allow_duplicate=True),
    # update na plots
    Output("figure_na_bar_plot", "figure", allow_duplicate=True),
    Output("figure_na_heatmap", "figure", allow_duplicate=True),
    # update overview page
    Output("datatable_overview", "data", allow_duplicate=True),
    Output("datatable_overview", "columns", allow_duplicate=True),
    # update sidebar
    Output("button_outlier", "disabled", allow_duplicate=True),
    Output("button_ts", "disabled", allow_duplicate=True),
    Output("button_sc", "disabled", allow_duplicate=True),
    Output("button_sr", "disabled", allow_duplicate=True),
    # inputs
    Input("button_na_apply", "n_clicks"),
    State("dropdown_na_feature", "value"),
    State("dropdown_na_method", "value"),
    State("dropdown_na_simple_strategy", "value"),
    State("input_na_simple_fill_value", "value"),   
    State("dropdown_na_iterative_filling_feature", "value"),
    State("slider_na_iterative_max_iter", "value"),
    State("dropdown_na_iterative_tol", "value"),
    State("slider_na_iterative_n_nearest_features", "value"),
    State("dropdown_na_iterative_initial_strategy", "value"),
    State("dropdown_na_iterative_fill_value", "value"),
    State("dropdown_na_iterative_imputation_order", "value"),
    State("dropdown_na_knn_filling_feature", "value"),
    State("slider_na_knn_n_neighbors", "value"),
    State("dropdown_na_knn_weights", "value"),
    State("input_na_manual_index", "value"),
    State("input_na_manual_fill_value", "value"),
    prevent_initial_call=True
)
def update_df_after_imputing(n_clicks, col, method, simple_strategy, simple_fill_value, iterative_filling_feature, iterative_max_iter, iterative_tol, iterative_n_nearest_features, iterative_initial_strategy, iterative_fill_value, iterative_imputation_order, knn_filling_feature, knn_n_neighbors, knn_weights, manual_index, manual_fill_value):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    # read out parameter
    params = {}
    if method == IMPUTER_METHODS[0]: # simple imputer
        params['missing_values'] = np.nan
        params['strategy'] = IMPUTER_STRATEGIES[simple_strategy]
        if simple_strategy == list(IMPUTER_STRATEGIES.keys())[3]:
            params['fill_value'] = simple_fill_value
    elif method == IMPUTER_METHODS[1]: # iterativ imputer
        col = [col] + iterative_filling_feature
        params['missing_values'] = np.nan
        params['max_iter'] = iterative_max_iter
        params['tol'] = iterative_tol
        if iterative_n_nearest_features < len(iterative_filling_feature):
            params['n_nearest_features'] = iterative_n_nearest_features
        else:
            params['n_nearest_features'] = None
        params['initial_strategy'] = IMPUTER_STRATEGIES[iterative_initial_strategy]
        if iterative_initial_strategy == list(IMPUTER_STRATEGIES.keys())[3]:
            params['fill_value'] = iterative_fill_value
        params['imputation_order'] = IMPUTER_ORDER[iterative_imputation_order]
    elif method == IMPUTER_METHODS[2]: # knn imputer
        col = [col] + knn_filling_feature
        params['missing_values'] = np.nan
        params['n_neighbors'] = knn_n_neighbors
        params['weights'] = IMPUTER_WEIGHTS[knn_weights]
    elif method == IMPUTER_METHODS[3]: # manual
        params['index'] = manual_index
        params['fill_value'] = manual_fill_value
    
    # apply imputing       
    tmp = table_data.DF_RAW.select_dtypes(include=NUMERICS)
    tmp = apply_imputing(tmp, col, method, params)
    for i in tmp.columns:
        table_data.DF_RAW[i] = tmp[i]    
    
    # remove helping columns
    if method == IMPUTER_METHODS[1] or method == IMPUTER_METHODS[2]:
        col = col[0]
            
    # update parameter
    df_num = table_data.DF_RAW.select_dtypes(include=NUMERICS)
    nan_cols = df_num.columns[df_num.isna().any()].tolist()
    df_na = table_data.DF_RAW[nan_cols]
    
    options = list(df_na.columns)
    
    if method == IMPUTER_METHODS[3]:
        value = col
    else:
        if len(options) > 0: 
            value = options[0]
        else:
            value = ""
        
    max_nearest_features = len(df_num.columns)
    marks = {i: {'label': str(round(i))} for i in np.arange(1, max_nearest_features, (max_nearest_features-1)/5)}
    if iterative_n_nearest_features > max_nearest_features:
        iterative_n_nearest_features = max_nearest_features
        
    
    # update na plots
    na_count = get_num_nan(table_data.DF_RAW)
    figure_na = get_na_bar_plot(na_count)
    figure_na_heatmap = get_na_heatmap(table_data.DF_RAW.isna())
    
    # update datatable
    df = analyse_df(table_data.DF_RAW)
    data_datatable_overview = [{col: df.loc[i, col] for col in df.columns} for i in df.index]
    columns_datatable_overview = [{"name": col, "id": col} for col in df.columns]   
    
    # update sidebar
    sidebar_disabled = nan_cols != []
    
    return options, value, max_nearest_features, iterative_n_nearest_features, marks, figure_na, figure_na_heatmap, data_datatable_overview, columns_datatable_overview, sidebar_disabled, sidebar_disabled, sidebar_disabled,  sidebar_disabled

@app.callback(
    Output("img_na_strategy", "src", allow_duplicate=True),  
    Output("link_na_strategy", "href", allow_duplicate=True),  
    Output("tooltip_na_strategy", "children", allow_duplicate=True), 
    Input("dropdown_na_method", "value"), 
    prevent_initial_call=True
)
def update_info(strategy):
    if strategy == IMPUTER_METHODS[0]:
        src = '/assets/img/link.png'
        href = IMPUTER_LINKS[0]
        children = IMPUTER_DESCRIPTIONS[0]
    elif strategy == IMPUTER_METHODS[1]:
        src = '/assets/img/link.png'
        href = IMPUTER_LINKS[1]
        children = IMPUTER_DESCRIPTIONS[1]
    elif strategy == IMPUTER_METHODS[2]:
        src = '/assets/img/link.png'
        href = IMPUTER_LINKS[2]
        children = IMPUTER_DESCRIPTIONS[2]
    elif strategy == IMPUTER_METHODS[3]:
        src = '/assets/img/tooltip.png'
        href = IMPUTER_LINKS[3]
        children = IMPUTER_DESCRIPTIONS[3]
    else:
        return dash.no_update

    return src, href, children