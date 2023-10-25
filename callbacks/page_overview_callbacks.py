# immport necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, ctx
import numpy as np

# import app
from view.app import app

# import data
from data import table_data

# import analyse and cleaning methods
from methods.cleaning import delete_columns
from methods.data_exploration.analyse import *

# import figures
from view.page_helper_components.plots import *

# import util
from methods.util import remove_item_if_exist, is_close

# import state management
from states.states import *

# import slider marks
from view.page_helper_components.sliders import get_slider_marks

# update after feature removal
@app.callback(
    # update textboard
    Output('text_board_shape', 'children', allow_duplicate=True),
    Output('text_board_memory', 'children', allow_duplicate=True),
    Output('text_board_na', 'children', allow_duplicate=True),
    Output('text_board_num', 'children', allow_duplicate=True),    
    # update histogram
    Output("dropdown_overview_features_selection_histogram", "options", allow_duplicate=True),
    Output("dropdown_overview_features_selection_histogram", "value", allow_duplicate=True), 
    Output("dropdown_overview_target_selection_histogram", "options", allow_duplicate=True),
    Output("dropdown_overview_target_selection_histogram", "value", allow_duplicate=True), 
    Output("dropdown_overview_class_selection_histogram", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_histogram", "value", allow_duplicate=True), 
    # update violinplot
    Output("dropdown_overview_features_selection_violinplot", "options", allow_duplicate=True),
    Output("dropdown_overview_features_selection_violinplot", "value", allow_duplicate=True), 
    Output("dropdown_overview_target_selection_violinplot", "options", allow_duplicate=True),
    Output("dropdown_overview_target_selection_violinplot", "value", allow_duplicate=True), 
    Output("dropdown_overview_class_selection_violinplot", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_violinplot", "value", allow_duplicate=True), 
    # update line plot
    Output("dropdown_overview_features_selection_linegraph", "options", allow_duplicate=True),
    Output("dropdown_overview_features_selection_linegraph", "value", allow_duplicate=True),  
    Output("dropdown_overview_target_selection_linegraph", "options", allow_duplicate=True),
    Output("dropdown_overview_target_selection_linegraph", "value", allow_duplicate=True), 
    Output("dropdown_overview_class_selection_linegraph", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_linegraph", "value", allow_duplicate=True), 
    Output("dropdown_overview_feature_selection_rangeslider_linegraph", "options", allow_duplicate=True),
    Output("dropdown_overview_feature_selection_rangeslider_linegraph", "value", allow_duplicate=True),  
    # update scatter plot
    Output("dropdown1_overview_feature_selection_scattergraph", "options", allow_duplicate=True),
    Output("dropdown1_overview_feature_selection_scattergraph", "value", allow_duplicate=True),
    Output("dropdown2_overview_feature_selection_scattergraph", "options", allow_duplicate=True),
    Output("dropdown2_overview_feature_selection_scattergraph", "value", allow_duplicate=True), 
    Output("dropdown_overview_target_selection_scattergraph", "options", allow_duplicate=True),
    Output("dropdown_overview_target_selection_scattergraph", "value", allow_duplicate=True), 
    Output("dropdown_overview_class_selection_scattergraph", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_scattergraph", "value", allow_duplicate=True), 
    # update correlation heatmap
    Output("dropdown_overview_feature_selection_heatmap", "options", allow_duplicate=True),
    Output("dropdown_overview_feature_selection_heatmap", "value", allow_duplicate=True),
    Output("dropdown_overview_target_selection_heatmap", "options", allow_duplicate=True),
    Output("dropdown_overview_target_selection_heatmap", "value", allow_duplicate=True), 
    Output("dropdown_overview_class_selection_heatmap", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_heatmap", "value", allow_duplicate=True), 
    # update categorical page
    Output("dropdown_categorical_feature", "options", allow_duplicate=True), 
    Output("dropdown_categorical_feature", "value", allow_duplicate=True), 
    Output("exploration_categorical_feature_ratio_bar_plot", "figure", allow_duplicate=True), 
    # update na page
    Output("figure_na_bar_plot", "figure", allow_duplicate=True), 
    Output("figure_na_heatmap", "figure", allow_duplicate=True), 
    Output("dropdown_na_feature", "options", allow_duplicate=True),
    Output("dropdown_na_feature", "value", allow_duplicate=True), 
    # update outlier page
    Output("figure_outlier_violin_plot", "figure", allow_duplicate=True),
    Output("table_outlier_detection", "data", allow_duplicate=True),
    Output("figure_outlier_preview", "figure", allow_duplicate=True),
    Output("dropdown_outlier_feature", "options", allow_duplicate=True),
    Output("dropdown_outlier_feature", "value", allow_duplicate=True),
    # update transformation
    Output("dropdown_transformation_time_series_dataset", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_dataset", "value", allow_duplicate=True),
    # update sidebar
    Output("button_categorical", 'disabled', allow_duplicate=True),
    Output("button_na_values", 'disabled', allow_duplicate=True),
    Output("button_outlier", 'disabled', allow_duplicate=True),
    Output("button_ts", 'disabled', allow_duplicate=True),
    Output("button_sc", 'disabled', allow_duplicate=True),
    Output("button_sr", 'disabled', allow_duplicate=True),
    # input
    State('datatable_overview', 'data_previous'),
    Input('datatable_overview', 'data'),    
    State("dropdown_overview_features_selection_histogram", "value"), 
    State("dropdown_overview_features_selection_linegraph", "value"), 
    State("dropdown_overview_feature_selection_rangeslider_linegraph", "value"),
    State("dropdown1_overview_feature_selection_scattergraph", "value"),
    State("dropdown2_overview_feature_selection_scattergraph", "value"),
    State("dropdown_overview_feature_selection_heatmap", "value"),
    State("dropdown_na_feature", "value"),
    prevent_initial_call=True
)
def delete_feature(previous, current, histogram_values, linegraph_values, linegraph_index_value, scatter1_value, scatter2_value, heatmap_value, na_value):   
    if table_data.DF_RAW is None:
        return dash.no_update
    
    row = None
    if previous is not None and len(previous) != len(current):
        row = previous[-1]['features']
        for pre, cur in zip(previous, current):
            if pre['features'] != cur['features']:
                row = pre['features']
                break
                        
        # update table
        table_data.DF_RAW = delete_columns(table_data.DF_RAW, row)
        
    # TODO delete later
    #table_data.DF_RAW.fillna(0, inplace=True)
        
    # update components
    df_num = table_data.DF_RAW.select_dtypes(include=NUMERICS)
        
    # update_options
    options_all = list(table_data.DF_RAW.columns)
    options_num = list(table_data.DF_RAW.select_dtypes(include=NUMERICS).columns)
    options_int = list(table_data.DF_RAW.select_dtypes(include=INTEGER).columns)
        
    
    # update textboard
    value_shape = html.H6(children=str(table_data.DF_RAW.shape), className='text_board_font1')
    value_memory =  html.H6(children=str(get_memory_usage(table_data.DF_RAW)), className='text_board_font1')
    value_na =  html.H6(children=str(get_percentage_nan_total(table_data.DF_RAW)), className='text_board_font1')
    value_num = html.H6(children=str(get_percentage_numeric(table_data.DF_RAW)), className='text_board_font1')
        
    # update histogram
    histogram_options = options_all
    histogram_target_options = options_all
    histogram_class_options = table_data.DF_RAW[options_all[0]].unique().tolist() + ['ALL']
    histogram_values = histogram_options[0]
    histogram_target_value = options_all[0]
    histogram_class_value = 'ALL'
    
    # update violinplot
    violinplot_options = options_all
    violinplot_target_options = options_all
    violinplot_class_options = table_data.DF_RAW[options_all[0]].unique().tolist() + ['ALL']
    violinplot_values = violinplot_options[:3]
    violinplot_target_value = violinplot_target_options[0]
    violinplot_class_value = 'ALL'
            
    # update linegraph
    linegraph_options = options_num
    linegraph_target_options = options_all
    linegraph_class_options = table_data.DF_RAW[options_all[0]].unique().tolist() + ['ALL']
    linegraph_index_options = ['index_auto'] + options_int 
    linegraph_values = linegraph_options[:3]
    linegraph_target_value = options_all[0]
    linegraph_class_value = 'ALL'
    linegraph_index_value = linegraph_index_options[0]
        
    # update scatter plot
    scatter_options = options_num
    scatter_target_options = options_all
    # Create an empty list to store column names with less than 50 unique values
    scatter_target_options = []

    # Loop through the columns in the DataFrame
    for column in table_data.DF_RAW.columns:
        unique_values = table_data.DF_RAW[column].nunique()
        if unique_values < 50:
            scatter_target_options.append(column)
    scatter_class_options = table_data.DF_RAW[options_all[0]].unique().tolist() + ['ALL']
    scatter1_value = options_num[0]
    scatter2_value = options_num[1]
    scatter_target_value = scatter_target_options[0]
    scatter_class_value = 'ALL'
            
    # update correlation heatmap  
    heat_map_options = options_num
    heat_map_target_options = options_all
    heat_map_class_options = table_data.DF_RAW[options_all[0]].unique().tolist() + ['ALL']
    heatmap_value = heat_map_options[:3]
    heat_map_target_value = options_all[0]
    heat_map_class_value = 'ALL'
        
    ### update categorical page 
    df_cat = table_data.DF_RAW.select_dtypes(include='object').dropna(axis=1, how='all') 
                          
    # update categorical dropdown
    options_cat = list(df_cat.columns)

    if len(options_cat) == 0:
        value_cat = dash.no_update
    else:
        value_cat = options_cat[0]
        
    # update ratio        
    num_num, num_cat = get_num_numeric_categorical(table_data.DF_RAW)    
    figure_cat = get_numeric_categorical_ratio_plot(num_num, num_cat)
        
    ### update na page
    # update parameter
    nan_cols = df_num.columns[df_num.isna().any()].tolist()
    df_na = table_data.DF_RAW[nan_cols]
    options_na = list(df_na.columns)
    
    if row is not None and na_value == row or (na_value is None and len(options_na) > 0):
        na_value = options_na[0]
    
    # update overview plots
    na_count = get_num_nan(table_data.DF_RAW)
    figure_num_na = get_na_bar_plot(na_count)
    figure_heatmap_na = get_na_heatmap(table_data.DF_RAW.isna())
    
    ### update outlier page
    # update violin distibution
    figure_violin = get_violin_plot(df_num, df_num.columns)
    
    # update features
    outlier_options = options_num
    outlier_value = options_num
    
    # update datatable
    data_outlier = []
    
    # update figure
    figure_outlier = {}
    
    # update state
    #save_dataset(table_data.DF_RAW, IN_PROCESSING_DATASETNAME, table_data.SEP)
    # TODO update outlier, transformation, supervised when no cat and no nan
    if len(list(table_data.ALL_DATASETS.keys())) > 0:
        #if CLEANED_DATASETNAME not in list(table_data.ALL_DATASETS.keys()):
        #    table_data.ALL_DATASETS[CLEANED_DATASETNAME] = table_data.DF_RAW
        #    table_data.ALL_RANGES[CLEANED_DATASETNAME] = [table_data.DF_RAW.index.min(), table_data.DF_RAW.index.max()]
        options_transformation = list(table_data.ALL_DATASETS.keys())
        value_transformation = options_transformation[0]
    else:
        options_transformation = []
        value_transformation = None
        
    # update sidebar
    cat_cols = table_data.DF_RAW.select_dtypes(include='object').columns
    nan_cols = table_data.DF_RAW.columns[table_data.DF_RAW.isna().any()].tolist()
        
    categorical_disabled = len(cat_cols) == 0
    na_disabled = len(nan_cols) == 0 or len(cat_cols) > 0
    rest_disabled = len(nan_cols) > 0 != [] or len(cat_cols) > 0
    
    training_disabled = True
        
        
    return value_shape, value_memory, value_na, value_num, histogram_options, histogram_values, histogram_target_options, histogram_target_value, histogram_class_options, histogram_class_value, violinplot_options, violinplot_values, violinplot_target_options, violinplot_target_value, violinplot_class_options, violinplot_class_value, linegraph_options, linegraph_values, linegraph_target_options, linegraph_target_value, linegraph_class_options, linegraph_class_value, linegraph_index_options, linegraph_index_value, scatter_options, scatter1_value, scatter_options, scatter2_value, scatter_target_options, scatter_target_value, scatter_class_options, scatter_class_value, heat_map_options, heatmap_value, heat_map_target_options, heat_map_target_value, heat_map_class_options, heat_map_class_value, options_cat, value_cat, figure_cat, figure_num_na, figure_heatmap_na, options_na, na_value, figure_violin, data_outlier, figure_outlier, outlier_options, outlier_value, options_transformation, value_transformation, categorical_disabled, na_disabled, rest_disabled, rest_disabled, training_disabled, training_disabled

    
# update the histogram and update the rangeslider in the histogram board according to the dropdown
@app.callback(
    Output("figure_overview_histogram", "children", allow_duplicate=True),
    Input("dropdown_overview_features_selection_histogram", "value"),        
    Input("dropdown_overview_target_selection_histogram", "value"),
    Input("dropdown_overview_class_selection_histogram", "value"),
    prevent_initial_call=True
)
def update_histogram_figure_under_constraint(cols, target, target_class):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    if target_class == 'ALL':
        target_class = None
        target = None
    
    df =  compute_plot(table_data.DF_RAW, None, cols, reset_index=True, target=target, target_class=target_class)
    # draw Figure
    figure =  get_overview_histogram_plot(df, cols)
    graph = dcc.Graph(figure=figure, className='figure_overview')
    
    return graph

# update the violinplot
@app.callback(
    Output("figure_overview_violinplot", "children", allow_duplicate=True),
    Input("dropdown_overview_features_selection_violinplot", "value"),        
    Input("dropdown_overview_target_selection_violinplot", "value"),
    Input("dropdown_overview_class_selection_violinplot", "value"),
    prevent_initial_call=True
)
def update_histogram_figure_under_constraint(cols, target, target_class):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    if target_class == 'ALL':
        target_class = None
        target = None
    
    df =  compute_plot(table_data.DF_RAW, None, cols, reset_index=True, target=target, target_class=target_class)
    # draw Figure
    figure =  get_overview_violin_plot(df, cols)
    graph = dcc.Graph(figure=figure ,className='figure_overview')
    
    return graph

# update the line plot and update the rangeslider in the line plot board according to the dropdown
@app.callback(
    Output("figure_overview_linegraph", "children", allow_duplicate=True),
    Input("dropdown_overview_features_selection_linegraph", "value"),
    Input("dropdown_overview_target_selection_linegraph", "value"),
    Input("dropdown_overview_class_selection_linegraph", "value"),
    Input("dropdown_overview_feature_selection_rangeslider_linegraph", "value"),
    prevent_initial_call=True
)
def update_line_plot_under_constraint(cols, target, target_class, col_index):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    if target_class == 'ALL':
        target_class = None
        target = None
        
    if col_index == 'index_auto':
        col_index = None
    
    df = compute_plot(table_data.DF_RAW, col_index, cols, reset_index=True, target=target, target_class=target_class)
    # draw Figure
    figure = get_overview_line_plot(df, cols)
    graph = dcc.Graph(figure=figure ,className='figure_overview')
    
    return graph

# update the scatter plot according to the dropdown
@app.callback(
    Output("figure_overview_scattergraph", "children", allow_duplicate=True),
    Input("dropdown1_overview_feature_selection_scattergraph", "value"),
    Input("dropdown2_overview_feature_selection_scattergraph", "value"),
    Input("dropdown_overview_target_selection_scattergraph", "value"),
    Input("dropdown_overview_class_selection_scattergraph", "value"),
    prevent_initial_call=True
)
def update_scatter_figure(col1, col2, target, target_class):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    df = compute_scatter(table_data.DF_RAW, col1, target=target, target_class=target_class)
    
    if target_class != 'ALL':
        target = None
    
    figure = get_overview_scatter_plot(df, col1, col2, target)
    graph = dcc.Graph(figure=figure ,className='figure_overview')
    
    return graph

# update heatmap
@app.callback(
    Output("figure_overview_heatmap", "figure", allow_duplicate=True), 
    Input("dropdown_overview_feature_selection_heatmap", "value"),
    Input("dropdown_overview_target_selection_heatmap", "value"),
    Input("dropdown_overview_class_selection_heatmap", "value"),
    prevent_initial_call=True
)
def update_heatmap(cols, target, target_class):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    if target_class == 'ALL':
        target_class = None
        target = None
    
    corr = analyse_correlation(table_data.DF_RAW, cols, target=target, target_class=target_class)
    figure = get_overview_heatmap(corr)
    return figure

# update class options histogram
@app.callback(
    Output("dropdown_overview_class_selection_histogram", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_histogram", "value", allow_duplicate=True),
    Input("dropdown_overview_target_selection_histogram", "value"),
    prevent_initial_call=True
)
def update_histogram(target):
    if target is None:
        return dash.no_update
    
    options = table_data.DF_RAW[target].unique().tolist() + ['ALL']
    options = sorted(options, key=lambda x: (isinstance(x, (int, float)), x))
    value = 'ALL'
    return options, value



# update class options linegraph
@app.callback(
    Output("dropdown_overview_class_selection_linegraph", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_linegraph", "value", allow_duplicate=True),
    Input("dropdown_overview_target_selection_linegraph", "value"),
    prevent_initial_call=True
)
def update_linegraph(target):
    if target is None:
        return dash.no_update
    
    options = table_data.DF_RAW[target].unique().tolist() + ['ALL']
    options = sorted(options, key=lambda x: (isinstance(x, (int, float)), x))
    value = 'ALL'
    return options, value

# update class options scatter
@app.callback(
    Output("dropdown_overview_class_selection_scattergraph", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_scattergraph", "value", allow_duplicate=True),
    Input("dropdown_overview_target_selection_scattergraph", "value"),
    prevent_initial_call=True
)
def update_scatter(target):
    if target is None:
        return dash.no_update
    
    options = table_data.DF_RAW[target].unique().tolist() + ['ALL']
    options = sorted(options, key=lambda x: (isinstance(x, (int, float)), x))
    value = 'ALL'
    return options, value

# update class options heatmap
@app.callback(
    Output("dropdown_overview_class_selection_heatmap", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_heatmap", "value", allow_duplicate=True),
    Input("dropdown_overview_target_selection_heatmap", "value"),
    prevent_initial_call=True
)
def update_heatmap(target):
    if target is None:
        return dash.no_update
    
    options = table_data.DF_RAW[target].unique().tolist() + ['ALL']
    options = sorted(options, key=lambda x: (isinstance(x, (int, float)), x))
    value = 'ALL'
    return options, value

# update class options violinplot
@app.callback(
    Output("dropdown_overview_class_selection_violinplot", "options", allow_duplicate=True),
    Output("dropdown_overview_class_selection_violinplot", "value", allow_duplicate=True),
    Input("dropdown_overview_target_selection_violinplot", "value"),
    prevent_initial_call=True
)
def update_violinplot(target):
    if target is None:
        return dash.no_update

    options = table_data.DF_RAW[target].unique().tolist() + ['ALL']
    options = sorted(options, key=lambda x: (isinstance(x, (int, float)), x))
    value = 'ALL'
    return options, value
