# Import necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, MATCH, ALL, ctx
import numpy as np
import json

# import app
from view.app import app

# import data
from data import table_data

# import methods
from methods.data_exploration.encoder import *
from methods.data_exploration.analyse import *
from methods.cleaning import delete_columns

# import figures
from view.page_helper_components.plots import get_numeric_categorical_ratio_plot, get_categorical_feature_pie_plot

# import utility
from methods.util import count_unique_values

@app.callback(
    Output("figure_categorical_feature_pie", "figure"),
    Output("container_feature_encoding", "style"),
    Input("dropdown_categorical_feature", "value"),
    State("container_feature_encoding", "style"),
)
def update_categorical_plot(col, style):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    triggered_id = ctx.triggered_id
    df_cat = table_data.DF_RAW.select_dtypes(include='object').dropna(axis=1, how='all') 
    
    if col is None or col == "" or triggered_id is None:
        return dash.no_update
    
    if col not in list(df_cat.columns):
        if style is None:
            style = {'display': 'none'} 
        else:
            style['display'] = 'none' 
        return dash.no_update, style
    
    counts = count_unique_values(df_cat, col)
    
    figure = get_categorical_feature_pie_plot(counts)
    
    return figure, style
    

@app.callback(
    # update categorical page
    Output("alert_categorical_unconvertable_string", "is_open"),
    Output("figure_categorical_feature_pie", "figure"),    
    Output("dropdown_categorical_feature", "options"),
    Output("dropdown_categorical_feature", "value"),
    Output("dropdown_replace_value1", "options"),    
    Output("dropdown_replace_value1", "value"),  
    Output("dropdown_replace_value2", "options"),
    Output("dropdown_replace_value2", "value"),
    # update overview page
    Output("datatable_overview", "data"),
    Output("datatable_overview", "columns"),
    # inputs
    Input("button_categorical_apply", "n_clicks"),
    State("dropdown_categorical_feature", "value"),
    State("dropdown_categorical_strategy", "value"),
    State("dropdown_replace_value1", "value"),   
    State("dropdown_replace_value2", "value"),
)
def update_df_after_encoding(n_clicks, col, strategy, in_str, out_str):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    triggered_id = ctx.triggered_id
    df_cat = table_data.DF_RAW.select_dtypes(include='object').dropna(axis=1, how='all') 
    
    if col is None or col == "" or triggered_id is None:
        return dash.no_update
    
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    ### update categorical page
    # apply encoding on df
    try:
        table_data.DF_RAW = apply_encoding(table_data.DF_RAW, col, strategy, in_str, out_str)
    except ValueError:
        print("Column seems to contain unconvertable string.")
        return [True] + 9 * [dash.no_update]
        
    
    df_cat = table_data.DF_RAW.select_dtypes(include='object').dropna(axis=1, how='all') 
    
    # update options
    options_cat = list(df_cat.columns)
    
    # hide feature encoding when no more categorical features otherwise update parameter
    if len(options_cat) > 0:     
        # if replacement selected update dropdowns
        if strategy == ENCODING_STRATEGIES[3]:
            unique_values = table_data.DF_RAW[col].unique().tolist()
    
            options_replacement = unique_values
            in_str = unique_values[0]
            
            out_str = unique_values[1]
            
            # don't update current feature
            value = col
            
            assert out_str in options_replacement
        else:
            options_replacement = dash.no_update
            in_str = dash.no_update
            
            # update features
            value = options_cat[0]
        
        # update pie plot
        counts = count_unique_values(df_cat, value)    
        figure = get_categorical_feature_pie_plot(counts)
    else:
            
        # don't update other components
        figure = dash.no_update
        options_cat = dash.no_update
        value = dash.no_update
        options_replacement = dash.no_update
        in_str = dash.no_update
        options_replacement = dash.no_update
        out_str = dash.no_update  
    
    ### update overview page
    
    # update datatable
    df = analyse_df(table_data.DF_RAW)
    data_datatable_overview = [{col: df.loc[i, col] for col in df.columns} for i in df.index]
    columns_datatable_overview = [{"name": col, "id": col} for col in df.columns]   
    
    
    return False, figure, options_cat, value, options_replacement, in_str, options_replacement, out_str, data_datatable_overview, columns_datatable_overview

@app.callback(
    Output("exploration_categorical_feature_ratio_bar_plot", "figure"),    
    Input("button_categorical_apply", "n_clicks"),
)
def update_ratio(n_clicks):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    num_num, num_cat = get_num_numeric_categorical(table_data.DF_RAW)
    
    figure = get_numeric_categorical_ratio_plot(num_num, num_cat)
    
    return figure

@app.callback(
    Output("dropdown_replace_value1", "options"),    
    Output("dropdown_replace_value1", "value"),  
    Output("dropdown_replace_value2", "options"),  
    Output("dropdown_replace_value2", "value"),  
    Input("dropdown_categorical_feature", "value"),
)
def update_replacement(col):
    triggered_id = ctx.triggered_id

    if col is None or col == "" or triggered_id is None:
        return dash.no_update
    
    unique_values = table_data.DF_RAW[col].unique().tolist()
    
    options = unique_values
    
    value1 = unique_values[0]
    value2 = unique_values[1]
    
    
    return options, value1, options, value2

@app.callback(
    Output("card_categorical_replacement", "style"),   
    Input("dropdown_categorical_strategy", "value"),
    State("card_categorical_replacement", "style"),   
)
def update_parameter(strategy, style):
    if strategy != ENCODING_STRATEGIES[3]:
        style['display'] = 'none'
    else:
        style['display'] = 'inline-block'
    
    return style



