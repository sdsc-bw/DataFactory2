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
    Output('text_board_shape', 'children'),
    Output('text_board_memory', 'children'),
    Output('text_board_na', 'children'),
    Output('text_board_num', 'children'),    
    # update histogram
    Output("dropdown_overview_features_selection_histogramgraph", "options"),
    Output("dropdown_overview_features_selection_histogramgraph", "value"),  
    Output("dropdown_overview_feature_selection_rangeslider_histogram", "options"),
    Output("dropdown_overview_feature_selection_rangeslider_histogram", "value"),  
    # update line plot
    Output("dropdown_overview_features_selection_linegraph", "options"),
    Output("dropdown_overview_features_selection_linegraph", "value"),  
    Output("dropdown_overview_feature_selection_rangeslider_linegraph", "options"),
    Output("dropdown_overview_feature_selection_rangeslider_linegraph", "value"),  
    # update scatter plot
    Output("dropdown1_overview_feature_selection_scattergraph", "options"),
    Output("dropdown1_overview_feature_selection_scattergraph", "value"),
    Output("dropdown2_overview_feature_selection_scattergraph", "options"),
    Output("dropdown2_overview_feature_selection_scattergraph", "value"),    
    # update correlation heatmap
    Output("dropdown_overview_feature_selection_heatmap", "options"),
    Output("dropdown_overview_feature_selection_heatmap", "value"),
    # update categorical page
    Output("dropdown_categorical_feature", "options"), 
    Output("dropdown_categorical_feature", "value"), 
    Output("exploration_categorical_feature_ratio_bar_plot", "figure"), 
    # update na page
    Output("figure_na_bar_plot", "figure"), 
    Output("figure_na_heatmap", "figure"), 
    Output("dropdown_na_feature", "options"),
    Output("dropdown_na_feature", "value"), 
    Output("slider_na_iterative_n_nearest_features", "max"),
    Output("slider_na_iterative_n_nearest_features", "value"),
    Output("slider_na_iterative_n_nearest_features", "marks"),
    # update outlier page
    Output("figure_outlier_violin_plot", "figure"),
    Output("table_outlier_detection", "data"),
    Output("figure_outlier_preview", "figure"),
    Output("dropdown_outlier_kv_feature", "options"),
    Output("dropdown_outlier_kv_feature", "value"),
    # update transformation
    Output("dropdown_transformation_time_series_dataset", "options"),
    Output("dropdown_transformation_time_series_dataset", "value"),
    # update sidebar
    Output("button_categorical", 'disabled'),
    Output("button_na_values", 'disabled'),
    Output("button_outlier", 'disabled'),
    Output("button_ts", 'disabled'),
    Output("button_sc", 'disabled'),
    Output("button_sr", 'disabled'),
    Output("button_usl", 'disabled'),
    # input
    State('datatable_overview', 'data_previous'),
    Input('datatable_overview', 'data'),    
    State("dropdown_overview_features_selection_histogramgraph", "value"), 
    State("dropdown_overview_feature_selection_rangeslider_histogram", "value"), 
    State("dropdown_overview_features_selection_linegraph", "value"), 
    State("dropdown_overview_feature_selection_rangeslider_linegraph", "value"),
    State("dropdown1_overview_feature_selection_scattergraph", "value"),
    State("dropdown2_overview_feature_selection_scattergraph", "value"),
    State("dropdown_overview_feature_selection_heatmap", "value"),
    State("dropdown_na_feature", "value"),
    State("slider_na_iterative_n_nearest_features", "value"),
    State("dropdown_outlier_kv_feature", "value"),
)
def delete_feature(previous, current, histogram_values, histogram_index_value, linegraph_values, linegraph_index_value, scatter1_value, scatter2_value, heatmap_value, na_value, iterative_n_nearest_features, outlier_kv_feature):   
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
        
    # update components
    df_num = table_data.DF_RAW.select_dtypes(include=NUMERICS)
        
    # update_options
    options_all = list(table_data.DF_RAW.columns)
    options_num = list(table_data.DF_RAW.select_dtypes(include=NUMERICS).columns)
    options_int = list(table_data.DF_RAW.select_dtypes(include=INTEGER).columns)
        
    
    # update textboard
    value_shape = str(table_data.DF_RAW.shape)
    value_memory = str(get_memory_usage(table_data.DF_RAW))
    value_na = str(get_percentage_nan_total(table_data.DF_RAW))
    value_num = str(get_percentage_numeric(table_data.DF_RAW))
        
    # update histogram
    histogram_options = options_all
    histogram_index_options = options_int
    histogram_values = histogram_options[:3]
    histogram_index_value = options_int[0]
            
    # update linegraph
    linegraph_options = options_all
    linegraph_index_options = options_int
    linegraph_values = linegraph_options[:3]
    linegraph_index_value = options_int[0]
        
    # update scatter plot
    scatter_options = options_num
    scatter1_value = options_num[0]
    scatter2_value = options_num[1]
            
    # update correlation heatmap  
    heat_map_options = options_num
    heatmap_value = heat_map_options[:3]
        
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
        
    max_nearest_features = len(df_num.columns)
    marks = {i: {'label': str(round(i))} for i in np.arange(1, max_nearest_features, (max_nearest_features-1)/5)}
    if iterative_n_nearest_features > max_nearest_features:
        iterative_n_nearest_features = max_nearest_features
    
    # update overview plots
    na_count = get_num_nan(table_data.DF_RAW)
    figure_num_na = get_na_bar_plot(na_count)
    figure_heatmap_na = get_na_heatmap(table_data.DF_RAW.isna())
    
    ### update outlier page
    # update violin distibution
    figure_violin = get_violin_plot(df_num, df_num.columns)
    
    # update datatable
    data_outlier = []
    if row is not None and outlier_kv_feature == row or outlier_kv_feature is None:
        outlier_kv_feature = options_num[0]
    
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
        
        
    return value_shape, value_memory, value_na, value_num, histogram_options, histogram_values, histogram_index_options, histogram_index_value, linegraph_options, linegraph_values, linegraph_index_options, linegraph_index_value, scatter_options, scatter1_value, scatter_options, scatter2_value, heat_map_options, heatmap_value, options_cat, value_cat, figure_cat, figure_num_na, figure_heatmap_na, options_na, na_value, max_nearest_features, iterative_n_nearest_features, marks, figure_violin, data_outlier, figure_outlier, options_num, outlier_kv_feature, options_transformation, value_transformation, categorical_disabled, na_disabled, rest_disabled, rest_disabled, training_disabled, training_disabled, training_disabled

    
# update the histogram and update the rangeslider in the histogram board according to the dropdown
@app.callback(
    [
        Output("figure_overview_histogram", "figure"),
        Output("rangeslider_overview_value_constraint_histogram", "min"),
        Output("rangeslider_overview_value_constraint_histogram", "max"),
        Output("rangeslider_overview_value_constraint_histogram", "marks"),
        Output("rangeslider_overview_value_constraint_histogram", "value"),
    ],
    [
        Input("dropdown_overview_features_selection_histogramgraph", "value"),        
        Input("dropdown_overview_features_selection_histogramgraph", "options"),
        Input("dropdown_overview_feature_selection_rangeslider_histogram", "value"),
        Input("rangeslider_overview_value_constraint_histogram", "value"),
        State("rangeslider_overview_value_constraint_histogram", "marks"),
        State("rangeslider_overview_value_constraint_histogram", "min"),
        State("rangeslider_overview_value_constraint_histogram", "max"),
    ]
)
def update_histogram_figure_under_constraint(cols, options, col_index, values, marks, curr_min, curr_max):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    triggered_id = ctx.triggered_id

    if triggered_id == "rangeslider_overview_value_constraint_histogram":
        value_min = values[0]
        value_max = values[1]
    else:
        value_min, value_max, curr_min, curr_max, _, values = get_marks_for_rangeslider(table_data.DF_RAW, col_index)
        
    marks = get_slider_marks((curr_min, curr_max))
    
    df =  compute_plot(table_data.DF_RAW, col_index, cols, value_min, value_max)
    # draw Figure
    figure =  get_overview_histogram_plot(df, cols)
    
    return figure, curr_min, curr_max, marks, values

# update the line plot and update the rangeslider in the line plot board according to the dropdown
@app.callback(
    [
        Output("figure_overview_linegraph", "figure"),
        Output("rangeslider_overview_value_constraint_linegraph", "min"),
        Output("rangeslider_overview_value_constraint_linegraph", "max"),
        Output("rangeslider_overview_value_constraint_linegraph", "marks"),
        Output("rangeslider_overview_value_constraint_linegraph", "value"),
    ],
    [
        Input("dropdown_overview_features_selection_linegraph", "value"),
        Input("dropdown_overview_features_selection_linegraph", "options"),
        Input("dropdown_overview_feature_selection_rangeslider_linegraph", "value"),
        Input("rangeslider_overview_value_constraint_linegraph", "value"),
        State("rangeslider_overview_value_constraint_linegraph", "marks"),
        State("rangeslider_overview_value_constraint_linegraph", "min"),
        State("rangeslider_overview_value_constraint_linegraph", "max"),
    ]
)
def update_line_plot_under_constraint(cols, options, col_index, values, marks, curr_min, curr_max):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    triggered_id = ctx.triggered_id

    if triggered_id == "rangeslider_overview_value_constraint_linegraph":
        value_min = values[0]
        value_max = values[1]
    else:
        value_min, value_max, curr_min, curr_max, _, values = get_marks_for_rangeslider(table_data.DF_RAW, col_index)
        
    marks = get_slider_marks((curr_min, curr_max))
    
    df = compute_plot(table_data.DF_RAW, col_index, cols, value_min, value_max, reset_index=True)
    # draw Figure
    figure = get_overview_line_plot(df, cols)
    
    return figure, curr_min, curr_max, marks, values

# update the scatter plot and update the rangeslider in the scatter plot board according to the dropdown
@app.callback(
    [
        Output("figure_overview_scattergraph", "figure"),
        Output("rangeslider_overview_value_constraint_scattergraph", "min"),
        Output("rangeslider_overview_value_constraint_scattergraph", "max"),
        Output("rangeslider_overview_value_constraint_scattergraph", "marks"),
        Output("rangeslider_overview_value_constraint_scattergraph", "value"),
    ],
    [
        Input("dropdown1_overview_feature_selection_scattergraph", "value"),
        Input("dropdown2_overview_feature_selection_scattergraph", "value"),
        Input("rangeslider_overview_value_constraint_scattergraph", "value"),
        State("rangeslider_overview_value_constraint_scattergraph", "marks"),
        State("rangeslider_overview_value_constraint_scattergraph", "min"),
        State("rangeslider_overview_value_constraint_scattergraph", "max"),
    ]
)
def update_scatter_figure_under_constraint(col1, col2, values, marks, curr_min, curr_max):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    triggered_id = ctx.triggered_id

    if triggered_id == "rangeslider_overview_value_constraint_scattergraph":
        value_min = values[0]
        value_max = values[1]
    else:
        value_min, value_max, curr_min, curr_max, _, values = get_marks_for_rangeslider(table_data.DF_RAW, col1)
        
    marks = get_slider_marks((curr_min, curr_max))
    
    df = compute_scatter(table_data.DF_RAW, col1, value_min, value_max)
    figure = get_overview_scatter_plot(df, col1, col2)
    
    return figure, curr_min, curr_max, marks, values

# update heatmap
@app.callback(
    Output("figure_overview_heatmap", "figure"),
    Input("dropdown_overview_feature_selection_heatmap", "value")
)
def update_heatmap(cols):
    if table_data.DF_RAW is None:
        return dash.no_update
    
    corr = analyse_correlation(table_data.DF_RAW[cols])
    figure = get_overview_heatmap(corr)
    return figure

