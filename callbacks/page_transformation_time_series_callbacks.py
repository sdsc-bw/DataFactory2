import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, MATCH, ALL, ctx

# import app
from view.app import app

# import data
from data import table_data

# import analyse and cleaning methods
from methods.cleaning import delete_columns
from methods.data_exploration.analyse import *

# import transformation methods
from methods.data_transformation.transformations_table_data import *

# import state management
from states.states import save_dataset_states, save_dataset

# import plots
from view.page_helper_components.plots import *

# import util
from methods.util import remove_item_if_exist

# import slider marks
from view.page_helper_components.sliders import get_slider_marks

# update datasets
@app.callback(
    Output("modal_transformation_time_series_delete_dataset", "is_open", allow_duplicate=True),
    Input("button_transformation_time_series_delete_dataset", "n_clicks"),     
    Input("button_transformation_time_series_delete_dataset_no", "n_clicks"),
    State("modal_transformation_time_series_delete_dataset", "is_open"),
    prevent_initial_call=True
)
def toggle_deletion_modal(n_clicks1, n_clicks2, is_open):
    if n_clicks1 is None or n_clicks1 == 0:
        return dash.no_update
    
    return not is_open

# update datasets
@app.callback(
    Output("button_transformation_time_series_save_dataset", "disabled"),
    Input("dropdown_transformation_time_series_dataset", "options"),
    prevent_initial_call=True
)
def disable_save_dataset(options):
    return not len(options) > 0

@app.callback(
    Output("modal_transformation_time_series_delete_dataset", "is_open", allow_duplicate=True),
    Output("dropdown_transformation_time_series_dataset", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_dataset", "value", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "value", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "value", allow_duplicate=True),
    Output("button_transformation_time_series_delete_dataset", "disabled", allow_duplicate=True),
    Input("button_transformation_time_series_delete_dataset_yes", "n_clicks"),
    State("modal_transformation_time_series_delete_dataset", "is_open"),
    State("dropdown_transformation_time_series_dataset", "value"),
    prevent_initial_call=True
)
def delete_datset(n_clicks, is_open, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    del table_data.ALL_DATASETS[dataset_name]
    
    # delete dataset in files
    #save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)

    # update datasets
    options_datasets = list(table_data.ALL_DATASETS.keys())
    curr_dataset = options_datasets[0]
    
    # update overview line plot
    df = table_data.ALL_DATASETS[curr_dataset]
    options_features = list(df.columns)
    value_overview = options_features[:4]
    
    # update parameter
    value_parameter = options_features[:3]
    
    # update disabled button
    disabled = len(list(table_data.ALL_DATASETS.keys())) < 2
    
    return not is_open, options_datasets, curr_dataset, options_features, value_overview, options_features, value_parameter, disabled

@app.callback(
    Output("modal_transformation_time_series_add_dataset", "is_open", allow_duplicate=True),
    Input("button_transformation_time_series_plus_dataset", "n_clicks"), 
    State("modal_transformation_time_series_add_dataset", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n_clicks, is_open):
    if n_clicks is None:
        return dash.no_update
    
    return not is_open

@app.callback(
    Output("dropdown_transformation_time_series_dataset", "value", allow_duplicate=True),
    Input("modal_transformation_time_series_add_dataset", "is_open"),
    prevent_initial_call=True
)
def close_modal(is_open):
    if is_open:
        return dash.no_update
    
    if len(list(table_data.ALL_DATASETS.keys())) == 0:
        return dash.no_update
    else:
        return list(table_data.ALL_DATASETS.keys())[-1]

@app.callback(
    Output("button_transformation_time_series_save_dataset", "n_clicks", allow_duplicate=True),
    Input("button_transformation_time_series_save_dataset", "n_clicks"), 
    State("dropdown_transformation_time_series_dataset", "value"),
    prevent_initial_call=True
)
def toggle_modal(n_clicks, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    #save_dataset(df, dataset_name, table_data.SEP)
    
    return dash.no_update

@app.callback(
    Output("alert_transformation_time_series_duplicate_dataset", "is_open", allow_duplicate=True),
    Output("modal_transformation_time_series_add_dataset", "is_open", allow_duplicate=True),
    Output("dropdown_transformation_time_series_dataset", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_dataset", "value", allow_duplicate=True),
    Output("input_transformation_time_series_new_dataset", "value", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "value", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "value", allow_duplicate=True),
    Output("button_transformation_time_series_delete_dataset", "disabled", allow_duplicate=True),
    Output("button_outlier", "disabled", allow_duplicate=True),
    Input("button_transformation_time_series_add_dataset", "n_clicks"),
    State("modal_transformation_time_series_add_dataset", "is_open"),
    State("input_transformation_time_series_new_dataset", "value"),
    prevent_initial_call=True
)
def add_dataset(n_clicks, is_open, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update

    if dataset_name in list(table_data.ALL_DATASETS.keys()):
        return [True] + 10 * [dash.no_update]
        
    is_open = not is_open
    
    # update datasets
    table_data.ALL_DATASETS[dataset_name] = table_data.DF_RAW.copy(deep=True)
    table_data.ALL_RANGES[dataset_name] = [table_data.DF_RAW.index.min(), table_data.DF_RAW.index.max()]
    table_data.ALL_MAX_RANGES[dataset_name] = [table_data.DF_RAW.index.min(), table_data.DF_RAW.index.max()]
    options_datasets = list(table_data.ALL_DATASETS.keys())
    curr_dataset = dataset_name
    
    # update overview line plot
    df = table_data.ALL_DATASETS[curr_dataset]
    options_features = list(df.columns)
    value_overview = options_features[:4]
    
    # update parameter
    value_parameter = options_features[:3]
    
    # update automatic dataset name
    new_dataset_name = 'new_dataset'

    # update disabled button
    disabled = len(list(table_data.ALL_DATASETS.keys())) < 2
    
    #save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)
    
    return False, is_open, options_datasets, dataset_name, new_dataset_name, options_features, value_overview, options_features, value_parameter, disabled, True

@app.callback(
    Output("alert_transformation_time_series_duplicate_feature_name", "is_open", allow_duplicate=True),
    Input("button_transformation_time_series_show", "n_clicks"),
    State("dropdown_transformation_time_series_feature_transformation", "value"),
    State("input_transformation_time_series_pca_feature_name", "value"),
    State("input_transformation_time_series_dwt_feature_name", "value"),
    State("dropdown_transformation_time_series_dataset", "value"),
    prevent_initial_call=True
)
def toggle_feature_alert(n_clicks, method, pca_feature_name, dwt_feature_name, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    if method == TRANSFORMATIONS_TS[2] and pca_feature_name in list(df.columns):
        return True
    if method == TRANSFORMATIONS_TS[3] and dwt_feature_name in list(df.columns):
        return True
    
    return False

@app.callback(
    Output("alert_transformation_time_series_polyorder", "is_open", allow_duplicate=True),
    Input("button_transformation_time_series_show", "n_clicks"),
    State("dropdown_transformation_time_series_feature_transformation", "value"),
    State("slider_transformation_time_series_sgf_polyorder", "value"),
    State("slider_transformation_time_series_sgf_periods", "value"),
    prevent_initial_call=True
)
def toggle_sgf_alert(n_clicks, method, polyorder, window_size):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    if method == TRANSFORMATIONS_TS[8] and polyorder >= window_size:
        return True
    
    return False

# update after feature removal
@app.callback(
    Output("dropdown_transformation_time_series_overview_feature", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "value", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "value", allow_duplicate=True),
    Output("dropdown_classification_target", "options", allow_duplicate=True),
    Output("dropdown_classification_target", "value", allow_duplicate=True),
    Output("dropdown_regression_target", "options", allow_duplicate=True),
    Output("dropdown_regression_target", "value", allow_duplicate=True),
    # input
    State('datatable_transformation_time_series_features', 'data_previous'),
    Input('datatable_transformation_time_series_features', 'data'),
    State("dropdown_transformation_time_series_dataset", "value"),
    State("dropdown_transformation_time_series_overview_feature", "value"),
    State("dropdown_transformation_time_series_features", "value"),
    State("dropdown_classification_dataset", "value"),
    State("dropdown_classification_target", "options"),
    State("dropdown_classification_target", "value"),
    State("dropdown_regression_dataset", "value"),
    State("dropdown_regression_target", "options"),
    State("dropdown_regression_target", "value"),
    prevent_initial_call=True
)
def delete_feature(previous, current, dataset_name, value_overview, value_parameter, value_dataset_name_classification, options_target_classification, value_target_classification, value_dataset_name_regression, options_target_regression, value_target_regression): 
    if dataset_name is None:
        return dash.no_update
    
    row = None
    if previous is not None and len(previous) != len(current):
        row = previous[-1]['Features']
        for pre, cur in zip(previous, current):
            if pre['Features'] != cur['Features']:
                row = pre['Features']
                break
                        
        # update table
        df = table_data.ALL_DATASETS[dataset_name]
        table_data.ALL_DATASETS[dataset_name] = delete_columns(df, row)
     
    df = table_data.ALL_DATASETS[dataset_name]
    options_features = list(df.columns)
    
    # update overview
    if row is not None and row in value_overview:
        remove_item_if_exist(value_overview, row)        
    if len(value_overview) == 0:
        value_overview = [options_features[0]]
    
    # update parameter
    if row is not None and row in value_parameter:
        remove_item_if_exist(value_parameter, row)        
    if len(value_parameter) == 0:
        value_parameter = [options_features[0]]
        
    # update classification
    if dataset_name == value_dataset_name_classification:
        options_target_classification = options_features
        value_target_classification = options_target_classification[0]
        
    # update regression
    if dataset_name == value_dataset_name_regression:
        options_target_regression = options_features
        value_target_regression = options_target_regression[0]
        
    # update states
    #save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)
        
    return options_features, value_overview, options_features, value_parameter, options_target_classification, value_target_classification, options_target_regression, value_target_regression

# update after selected dataset changes
@app.callback(
    Output('datatable_transformation_time_series_features', 'data', allow_duplicate=True),
    Output('datatable_transformation_time_series_features', 'data_previous', allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "value", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "value", allow_duplicate=True),
    Output('checklist_transformation_time_series_all_features', 'value', allow_duplicate=True),    
    Output("rangeslider_transformation_time_series_overview", "value", allow_duplicate=True),
    Output("rangeslider_transformation_time_series_overview", "min", allow_duplicate=True),
    Output("rangeslider_transformation_time_series_overview", "max", allow_duplicate=True),
    Output("rangeslider_transformation_time_series_overview", "marks", allow_duplicate=True),
    Input("dropdown_transformation_time_series_dataset", "value"),
    prevent_initial_call=True
)
def update_after_dataset_changes(dataset_name):
    if dataset_name is None or dataset_name == '':
        return dash.no_update

    df = table_data.ALL_DATASETS[dataset_name]
    
    # update datatable
    data = pd.DataFrame({'Features': df.columns}).to_dict('records')
    
    # update overview line plot
    df = table_data.ALL_DATASETS[dataset_name]
    options_features = list(df.columns)
    value_overview = options_features[:4]
    
    #range_values =  table_data.ALL_RANGES[dataset_name]
    
    # update rangeslider
    range_values = table_data.ALL_RANGES[dataset_name]
    range_min, range_max =  table_data.ALL_MAX_RANGES[dataset_name]
    
    marks = get_slider_marks((range_min, range_max))
    
    # update parameter
    checklist_value = []
    value_parameter = options_features[:3]    
    
    return data, None, options_features, value_overview, options_features, value_parameter, checklist_value, range_values, range_min, range_max, marks

# update selected features
@app.callback(
    Output('dropdown_transformation_time_series_features', 'value', allow_duplicate=True),
    Input("checklist_transformation_time_series_all_features", "value"),
    State('dropdown_transformation_time_series_features', 'options'),
    prevent_initial_call=True
)
def update_selected_features(all_features, options):
    if all_features is None or all_features == []:
        return dash.no_update
    
    return options

# update selected features when plot changes
@app.callback(
    Output('dropdown_transformation_time_series_overview_feature', 'value', allow_duplicate=True),
    Input("dropdown_transformation_time_series_plots", "value"),
    Input('dropdown_transformation_time_series_overview_feature', 'value'),
    prevent_initial_call=True
)
def update_selected_features_after_plot_changes(plot, cols):
    if cols is None or cols == []:
        return dash.no_update
    
    if plot == PLOTS[1]:
        # histogram only needs one feature
        cols = [cols[0]]
    elif plot == PLOTS[3]:
        if len(cols) > 2:
            cols = cols[:2]
    else:
        return dash.no_update
    
    return cols

# update after selected dataset changes
@app.callback(
    Output("dropdown_classification_dataset", "options", allow_duplicate=True),
    Output("dropdown_classification_dataset", "value", allow_duplicate=True),    
    Output("dropdown_regression_dataset", "options", allow_duplicate=True),
    Output("dropdown_regression_dataset", "value", allow_duplicate=True), 
    Input("dropdown_transformation_time_series_dataset", "options"),
    State("dropdown_classification_dataset", "value"), 
    State("dropdown_regression_dataset", "value"),
    prevent_initial_call=True
)
def update_after_dataset_changes(datasets, dataset_name_classification, dataset_name_regression):
    if datasets is None or datasets == []:
        return dash.no_update
    
    
    options = list(table_data.ALL_DATASETS.keys())
    if dataset_name_classification not in options:
        dataset_name_classification = options[0]
        
    if dataset_name_regression not in options:
        dataset_name_regression = options[0]
        
    df = table_data.ALL_DATASETS[dataset_name_classification]
    
    # update rangeslider
    #min_range = df.index.min() 
    #max_range = df.index.max()
    #value_range = table_data.ALL_RANGES[dataset_name_classification]
    #range_min, range_max =  table_data.ALL_MAX_RANGES[dataset_name]
    
    #marks = get_slider_marks((min_range, max_range))
    
    return options, dataset_name_classification, options, dataset_name_regression

# update overview plot
@app.callback(
    Output("figure_transformation_time_series_overview", "figure", allow_duplicate=True),
    Input("dropdown_transformation_time_series_overview_feature", "options"),
    Input("dropdown_transformation_time_series_overview_feature", "value"),
    Input("dropdown_transformation_time_series_plots", "value"),
    State("dropdown_transformation_time_series_dataset", "value"),
    Input("rangeslider_transformation_time_series_overview", "value"),
    Input("rangeslider_transformation_time_series_overview", "min"),
    Input("rangeslider_transformation_time_series_overview", "max"),
    prevent_initial_call=True
)
def update_overview_plot(options, cols, plot, dataset_name, values_range, range_min, range_max):
    if cols is None or cols == "" or cols == []:
        return dash.no_update
    if dataset_name is None or dataset_name == '':
        return dash.no_update

    value_min = values_range[0]
    value_max = values_range[1]

    # update graph if featues change
    for c in cols:
        if c not in options:
            cols = options[:3]
    
    df = table_data.ALL_DATASETS[dataset_name].loc[value_min:value_max]
    if plot == PLOTS[0]:
        df = compute_plot(df, None, cols, value_min, value_max, reset_index=True)
        figure = get_overview_line_plot(df, cols)
    elif plot == PLOTS[1]:
        df = compute_plot(df, None, cols, value_min, value_max, reset_index=True)
        figure = get_overview_histogram_plot(df, cols)
    elif plot == PLOTS[2]:
        corr = analyse_correlation(df, cols)
        figure = get_overview_heatmap(corr)
    elif plot == PLOTS[3]:
        if len(cols) >= 2:
            x = cols[0]
            y = cols[1]
        elif len(cols) == 1:
            x = cols[0]
            y = cols[0]
        else:
            return dash.no_update
        figure = get_overview_scatter_plot(df, x, y) 
    elif plot == PLOTS[4]:
         figure = get_violin_plot(df, cols, max_index=None)
    
    # save range
    triggered_id = ctx.triggered_id
    if triggered_id == "rangeslider_transformation_time_series_overview":
        table_data.ALL_RANGES[dataset_name] = values_range
        table_data.ALL_MAX_RANGES[dataset_name] = (range_min, range_max)
    #save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)
    
    return figure

# update parameter
@app.callback(
    Output("container_transformation_time_series_pca", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    State("container_transformation_time_series_pca", "style"),
    prevent_initial_call=True
)
def update_style_pca(method, style):
    if style is None:
        style = {}
    if method == TRANSFORMATIONS_TS[2]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("slider_transformation_time_series_pca_n_components", "max", allow_duplicate=True),
    Output("input_transformation_time_series_pca_feature_name", "value", allow_duplicate=True),
    Input("dropdown_transformation_time_series_features", "value"),
    prevent_initial_call=True
)
def update_style_pca_components(value):
    if value is None or value == "":
        return dash.no_update
    
    max_n_components = len(value) - 1
    if max_n_components < 1:
        max_n_components = 1
    
    feature_name = '_'.join(value)
    
    return max_n_components, feature_name

@app.callback(
    Output("input_transformation_time_series_dwt_feature_name", "value", allow_duplicate=True),
    Input("dropdown_transformation_time_series_features", "value"),
    prevent_initial_call=True
)
def update_style_pca_components(value):
    if value is None or value == "":
        return dash.no_update
    
    feature_name = '_'.join(value)
    
    return feature_name

@app.callback(
    Output("slider_transformation_time_series_dwt_vanishing_moments", "min", allow_duplicate=True),
    Input("dropdown_transformation_time_series_dwt_wavelet", "value"),
    prevent_initial_call=True
)
def update_style_dwt_n(wavelet):
    if wavelet == list(WAVELETS.keys())[1]:
        min_n = 2
    else:
        min_n = 1
    
    return min_n

@app.callback(
    Output("container_transformation_time_series_dwt", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    State("container_transformation_time_series_dwt", "style"),
    prevent_initial_call=True
)
def update_style_dwt(method, style):
    if style is None:
        style = {}
        
    if method == TRANSFORMATIONS_TS[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_transformation_time_series_dwt_vanishing_moments", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    Input("dropdown_transformation_time_series_dwt_wavelet", "value"),
    State("container_transformation_time_series_dwt_vanishing_moments", "style"),
    prevent_initial_call=True
)
def update_style_dwt_parameter(method, wavelet, style):
    if style is None:
        style = {}
    if method == TRANSFORMATIONS_TS[3] and wavelet != list(WAVELETS.keys())[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_transformation_time_series_shift", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    State("container_transformation_time_series_shift", "style"),
    prevent_initial_call=True
)
def update_style_shifting(method, style):
    if style is None:
        style = {}
        
    if method == TRANSFORMATIONS_TS[5]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_transformation_time_series_sw", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    State("container_transformation_time_series_sw", "style"),
    prevent_initial_call=True
)
def update_style_sliding_window(method, style):
    if style is None:
        style = {}
        
    if method == TRANSFORMATIONS_TS[6]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_transformation_time_series_diff", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    State("container_transformation_time_series_diff", "style"),
    prevent_initial_call=True
)
def update_style_differencing(method, style):
    if style is None:
        style = {}
        
    if method == TRANSFORMATIONS_TS[7]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_transformation_time_series_sgf", "style", allow_duplicate=True),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    State("container_transformation_time_series_sgf", "style"),
    prevent_initial_call=True
)
def update_style_differencing(method, style):
    if style is None:
        style = {}
        
    if method == TRANSFORMATIONS_TS[8]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

# update parameter
@app.callback(
    Output("slider_transformation_time_series_sgf_periods", "value", allow_duplicate=True),
    Input("slider_transformation_time_series_sgf_polyorder", "value"),
    State("slider_transformation_time_series_sgf_periods", "value"),
    prevent_initial_call=True
)
def update_sgf_periods(polyorder, periods):
    if polyorder is None or periods is None:
        return dash.no_update
    
    # polyorder must be less than periods
    if polyorder < periods:
        return periods
    else: 
        if (polyorder + 1) % 2 == 1:
            return polyorder + 1
        else:
            return polyorder + 2
        
# update parameter
@app.callback(
    Output("slider_transformation_time_series_sgf_polyorder", "value", allow_duplicate=True),
    State("slider_transformation_time_series_sgf_polyorder", "value"),
    Input("slider_transformation_time_series_sgf_periods", "value"),
    prevent_initial_call=True
)
def update_sgf_polyorder(polyorder, periods):
    if polyorder is None or periods is None:
        return dash.no_update
    
    # polyorder must be less than periods
    if polyorder < periods:
        return polyorder
    else: 
        return periods - 1


# update button styles
@app.callback(
    Output("button_transformation_time_series_apply", "style", allow_duplicate=True),
    Output("button_transformation_time_series_show", "style", allow_duplicate=True),
    Input("button_transformation_time_series_show", "n_clicks"),
    Input("dropdown_transformation_time_series_dataset", "value"),
    Input("dropdown_transformation_time_series_features", "value"),
    Input("dropdown_transformation_time_series_feature_transformation", "value"),
    Input("slider_transformation_time_series_pca_n_components", "value"),
    Input("input_transformation_time_series_pca_feature_name", "value"),
    Input("input_transformation_time_series_dwt_feature_name", "value"),
    Input("dropdown_transformation_time_series_dwt_wavelet", "value"),
    Input("slider_transformation_time_series_dwt_mode", "value"),
    Input("slider_transformation_time_series_dwt_level", "value"),
    Input("slider_transformation_time_series_dwt_vanishing_moments", "value"),
    Input('datatable_transformation_time_series_features', 'data'),
    Input('datatable_transformation_time_series_features', 'data_previous'),
    Input('slider_transformation_time_series_shift_steps', 'value'),
    Input('checklist_transformation_time_series_shift_multi', 'value'),
    Input('dropdown_transformation_time_series_sw_operations', 'value'),
    Input('slider_transformation_time_series_sw_periods', 'value'),
    Input('slider_transformation_time_series_diff_periods', 'value'),
    Input('slider_transformation_time_series_sgf_polyorder', 'value'),
    Input('slider_transformation_time_series_sgf_periods', 'value'),
    State("button_transformation_time_series_apply", "style"),
    State("button_transformation_time_series_show", "style"),
    prevent_initial_call=True
)
def update_style_buttons(n_clicks, dataset_name, features, method, v4, pca_feature_name, dwt_feature_name, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, sgf_poly_order, sgf_periods, style_apply, style_show):  
    if style_apply is None:
        style_apply = {}
    if style_show is None:
        style_show = {}   
    
    if len(table_data.ALL_DATASETS.keys()) == 0:
        style_show['display'] = 'block'
        style_apply['display'] = 'none' 
        return style_apply, style_show
    
    # handle false input
    if dataset_name is None:
        return dash.no_update
    
    if method == TRANSFORMATIONS_TS[8] and sgf_poly_order >= sgf_periods:
        return dash.no_update
        
    df = table_data.ALL_DATASETS[dataset_name]
    
    remaining_features = list(set(df.columns) - set(features))
    
    if method == TRANSFORMATIONS_TS[2] and pca_feature_name in remaining_features:
        return dash.no_update
    
    # change button style
    triggered_id = ctx.triggered_id
    
    if triggered_id is None or triggered_id == 'button_transformation_time_series_show':
        style_apply['display'] = 'block'
        style_show['display'] = 'none'   
    else:
        style_show['display'] = 'block'
        style_apply['display'] = 'none' 
        
    return style_apply, style_show


# update line plot
@app.callback(
    # update preview plot
    Output("loading_transformation_time_series_preview", "children", allow_duplicate=True),
    # inputs
    Input("button_transformation_time_series_show", "n_clicks"),
    State("dropdown_transformation_time_series_dataset", "value"),
    State("dropdown_transformation_time_series_features", "value"),
    State("dropdown_transformation_time_series_feature_transformation", "value"),
    State("slider_transformation_time_series_pca_n_components", "value"),
    State("input_transformation_time_series_pca_feature_name", "value"),
    State("dropdown_transformation_time_series_dwt_wavelet", "value"),
    State("slider_transformation_time_series_dwt_mode", "value"),
    State("slider_transformation_time_series_dwt_level", "value"),
    State("slider_transformation_time_series_dwt_vanishing_moments", "value"),
    State("input_transformation_time_series_dwt_feature_name", "value"),
    State('slider_transformation_time_series_shift_steps', 'value'),
    State('checklist_transformation_time_series_shift_multi', 'value'),
    State('dropdown_transformation_time_series_sw_operations', 'value'),
    State('slider_transformation_time_series_sw_periods', 'value'),
    State('slider_transformation_time_series_diff_periods', 'value'),
    State('slider_transformation_time_series_sgf_polyorder', 'value'),
    State('slider_transformation_time_series_sgf_periods', 'value'),
    prevent_initial_call=True
)
def update_preview_plot(n_clicks, dataset_name, cols, method, pca_n_components, pca_feature_name, dwt_wavelet, dwt_mode, dwt_level, dwt_vanishing_moments, dwt_feature_name, shift_steps, shift_multi, sw_operations, sw_periods, diff_periods, sgf_poly_order, sgf_periods):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    # handle false input 
    #if method == TRANSFORMATIONS_TS[2] and pca_feature_name in list(df.columns):
    #    return dash.no_update
    
    if method == TRANSFORMATIONS_TS[8] and sgf_poly_order >= sgf_periods:
        return dash.no_update
    
    # read out parameter
    params = {}
    if method == TRANSFORMATIONS_TS[2]: # pca
        params['n_components'] = pca_n_components
        params['feature_name'] = pca_feature_name
    elif method == TRANSFORMATIONS_TS[3]: # dwt       
        params['mode'] = WAVELET_MODES[dwt_mode]
        params['level'] = dwt_level
        params['wavelet'] = WAVELETS[dwt_wavelet]
        params['feature_name'] = dwt_feature_name
        if dwt_wavelet != list(WAVELETS.keys())[3]:
            params['n'] = dwt_vanishing_moments
    elif method == TRANSFORMATIONS_TS[5]: # shift      
        params['shift_steps'] = shift_steps
        params['multi_shift'] = shift_multi != []
    elif method == TRANSFORMATIONS_TS[6]: # sliding window      
        params['window_size'] = sw_periods
        params['operation'] = SLIDING_WINDOW_OPERATIONS[sw_operations]
    elif method == TRANSFORMATIONS_TS[7]: # differencing 
        params['window_size'] = diff_periods
    elif method == TRANSFORMATIONS_TS[8]: # savitzky golay filter 
        params['window_size'] = sgf_periods
        params['polyorder'] = sgf_poly_order
    
    # apply transformation
    df, cols = apply_transformation_time_series(df, cols, method, params)

    # update line plot preview  
    figure = get_overview_line_plot(df, cols, index=None)
    graph = dcc.Graph(id="figure_transformation_time_series_preview", className='graph_categorical', figure=figure)
    
    return graph

# update df and overview
@app.callback(
    # update preview plot
    Output('datatable_transformation_time_series_features', 'data', allow_duplicate=True),
    Output('datatable_transformation_time_series_features', 'data_previous', allow_duplicate=True),
    # update parameter
    Output("dropdown_transformation_time_series_features", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_features", "value", allow_duplicate=True),
    # update overview
    Output("dropdown_transformation_time_series_overview_feature", "options", allow_duplicate=True),
    Output("dropdown_transformation_time_series_overview_feature", "value", allow_duplicate=True),
    # inputs
    Input("button_transformation_time_series_apply", "n_clicks"),
    State("dropdown_transformation_time_series_dataset", "value"),
    State("dropdown_transformation_time_series_features", "value"),
    State("dropdown_transformation_time_series_feature_transformation", "value"),
    State("slider_transformation_time_series_pca_n_components", "value"),
    State("input_transformation_time_series_pca_feature_name", "value"),
    State("dropdown_transformation_time_series_dwt_wavelet", "value"),
    State("slider_transformation_time_series_dwt_mode", "value"),
    State("slider_transformation_time_series_dwt_level", "value"),
    State("slider_transformation_time_series_dwt_vanishing_moments", "value"),
    State("input_transformation_time_series_dwt_feature_name", "value"),
    State('slider_transformation_time_series_shift_steps', 'value'),
    State('checklist_transformation_time_series_shift_multi', 'value'),
    State('dropdown_transformation_time_series_sw_operations', 'value'),
    State('slider_transformation_time_series_sw_periods', 'value'),
    State('slider_transformation_time_series_diff_periods', 'value'),
    State('slider_transformation_time_series_sgf_polyorder', 'value'),
    State('slider_transformation_time_series_sgf_periods', 'value'),
    State("dropdown_transformation_time_series_overview_feature", "value"),
    prevent_initial_call=True
)
def update_overview(n_clicks, dataset_name, cols, method, pca_n_components, pca_feature_name, dwt_wavelet, dwt_mode, dwt_level, dwt_vanishing_moments, dwt_feature_name, shift_steps, shift_multi, sw_operations, sw_periods, diff_periods, sgf_poly_order, sgf_periods, value_overview):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    # read out parameter
    params = {}
    if method == TRANSFORMATIONS_TS[2]: # pca
        params['n_components'] = pca_n_components
        params['feature_name'] = pca_feature_name
    elif method == TRANSFORMATIONS_TS[3]: # dwt       
        params['mode'] = WAVELET_MODES[dwt_mode]
        params['level'] = dwt_level
        params['wavelet'] = WAVELETS[dwt_wavelet]
        params['feature_name'] = dwt_feature_name
        if dwt_wavelet != list(WAVELETS.keys())[3]:
            params['n'] = dwt_vanishing_moments
    elif method == TRANSFORMATIONS_TS[5]: # shift      
        params['shift_steps'] = shift_steps
        params['multi_shift'] = shift_multi != []
    elif method == TRANSFORMATIONS_TS[6]: # sliding window      
        params['window_size'] = sw_periods
        params['operation'] = SLIDING_WINDOW_OPERATIONS[sw_operations]
    elif method == TRANSFORMATIONS_TS[7]: # differencing 
        params['window_size'] = diff_periods
    elif method == TRANSFORMATIONS_TS[8]: # savitzky golay filter 
        params['window_size'] = sgf_periods
        params['polyorder'] = sgf_poly_order
    
    # apply transformation
    table_data.ALL_DATASETS[dataset_name], _ = apply_transformation_time_series(table_data.ALL_DATASETS[dataset_name], cols, method, params)
    
    # update overview
    options_features = list(table_data.ALL_DATASETS[dataset_name].columns)
    if method == TRANSFORMATIONS_TS[2]:
        for col in cols:
            remove_item_if_exist(value_overview, col)        
    if len(value_overview) == 0:
        value_overview = [options_features[0]]
        
    # update paramter
    value_parameter = options_features[:3]
    
    # save changes
    #save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)

    # update feature table
    data = pd.DataFrame({'Features': table_data.ALL_DATASETS[dataset_name].columns}).to_dict('records')
    
    return data, None, options_features, value_parameter, options_features, value_overview

@app.callback(
    Output("img_time_series_strategy", "src", allow_duplicate=True),  
    Output("link_time_series_strategy", "href", allow_duplicate=True),  
    Output("tooltip_time_series_strategy", "children", allow_duplicate=True), 
    Input("dropdown_transformation_time_series_feature_transformation", "value"), 
    prevent_initial_call=True
)
def update_info(strategy):
    if strategy == TRANSFORMATIONS_TS[0]:
        src = '/assets/img/tooltip.png'
        href = TRANSFORMATIONS_LINKS[0]
        children = TRANSFORMATIONS_DESCRIPTIONS[0]
    elif strategy == TRANSFORMATIONS_TS[1]:
        src = '/assets/img/tooltip.png'
        href = TRANSFORMATIONS_LINKS[1]
        children = TRANSFORMATIONS_DESCRIPTIONS[1]
    elif strategy == TRANSFORMATIONS_TS[2]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[2]
        children = TRANSFORMATIONS_DESCRIPTIONS[2]
    elif strategy == TRANSFORMATIONS_TS[3]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[3]
        children = TRANSFORMATIONS_DESCRIPTIONS[3]
    elif strategy == TRANSFORMATIONS_TS[4]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[4]
        children = TRANSFORMATIONS_DESCRIPTIONS[4]
    elif strategy == TRANSFORMATIONS_TS[5]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[5]
        children = TRANSFORMATIONS_DESCRIPTIONS[5]
    elif strategy == TRANSFORMATIONS_TS[6]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[6]
        children = TRANSFORMATIONS_DESCRIPTIONS[6]
    elif strategy == TRANSFORMATIONS_TS[7]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[7]
        children = TRANSFORMATIONS_DESCRIPTIONS[7]
    elif strategy == TRANSFORMATIONS_TS[8]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[8]
        children = TRANSFORMATIONS_DESCRIPTIONS[8]
    elif strategy == TRANSFORMATIONS_TS[9]:
        src = '/assets/img/link.png'
        href = TRANSFORMATIONS_LINKS[9]
        children = TRANSFORMATIONS_DESCRIPTIONS[9]
    else:
        return dash.no_update

    return src, href, children
