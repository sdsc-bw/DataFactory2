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

# update datasets
@app.callback(
    Output("modal_transformation_table_data_delete_dataset", "is_open"),
    Input("button_delete_dataset", "n_clicks"),     
    Input("button_transformation_table_data_delete_dataset_no", "n_clicks"),
    State("modal_transformation_table_data_delete_dataset", "is_open"),
)
def toggle_deletion_modal(n_clicks1, n_clicks2, is_open):
    if n_clicks1 is None or n_clicks1 == 0:
        return dash.no_update
    
    return not is_open

@app.callback(
    Output("modal_transformation_table_data_delete_dataset", "is_open"),
    Output("dropdown_transformation_table_data_dataset", "options"),
    Output("dropdown_transformation_table_data_dataset", "value"),
    Output("dropdown_transformation_table_data_overview_feature", "options"),
    Output("dropdown_transformation_table_data_overview_feature", "value"),
    Output("dropdown_transformation_table_data_features", "options"),
    Output("dropdown_transformation_table_data_features", "value"),
    Output("button_delete_dataset", "disabled"),
    Input("button_transformation_table_data_delete_dataset_yes", "n_clicks"),
    State("modal_transformation_table_data_delete_dataset", "is_open"),
    State("dropdown_transformation_table_data_dataset", "value"),
)
def delete_datset(n_clicks, is_open, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    del table_data.ALL_DATASETS[dataset_name]
    
    # delete dataset in files
    save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)

    # update datasets
    options_datasets = list(table_data.ALL_DATASETS.keys()) + ['New Dataset...']
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
    Output("modal_transformation_table_data_add_dataset", "is_open"),
    Input("dropdown_transformation_table_data_dataset", "value"), 
    State("modal_transformation_table_data_add_dataset", "is_open"),
)
def toggle_modal(value, is_open):
    if value is None:
        return dash.no_update
    
    if value in list(table_data.ALL_DATASETS.keys()):
        return dash.no_update
    
    return not is_open

@app.callback(
    Output("button_save_dataset", "n_clicks"),
    Input("button_save_dataset", "n_clicks"), 
    State("dropdown_transformation_table_data_dataset", "value"),
)
def toggle_modal(n_clicks, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    save_dataset(df, dataset_name, table_data.SEP)
    
    return dash.no_update

@app.callback(
    Output("modal_transformation_table_data_add_dataset", "is_open"),
    Output("dropdown_transformation_table_data_dataset", "options"),
    Output("dropdown_transformation_table_data_dataset", "value"),
    Output("input_transformation_table_data_new_dataset", "value"),
    Output("dropdown_transformation_table_data_overview_feature", "options"),
    Output("dropdown_transformation_table_data_overview_feature", "value"),
    Output("dropdown_transformation_table_data_features", "options"),
    Output("dropdown_transformation_table_data_features", "value"),
    Output("button_delete_dataset", "disabled"),
    Input("button_transformation_table_data_add_dataset", "n_clicks"),
    State("modal_transformation_table_data_add_dataset", "is_open"),
    State("input_transformation_table_data_new_dataset", "value"),
)
def add_dataset(n_clicks, is_open, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update

    if dataset_name in list(table_data.ALL_DATASETS.keys()):
        return dash.no_update
        
    is_open = not is_open
    
    # update datasets
    table_data.ALL_DATASETS[dataset_name] = table_data.DF_RAW.copy(deep=True)
    table_data.ALL_RANGES[dataset_name] = [table_data.DF_RAW.index.min(), table_data.DF_RAW.index.max()]
    options_datasets = list(table_data.ALL_DATASETS.keys()) + ['New Dataset...']
    curr_dataset = dataset_name
    
    # update overview line plot
    df = table_data.ALL_DATASETS[curr_dataset]
    options_features = list(df.columns)
    value_overview = options_features[:4]
    
    # update parameter
    value_parameter = options_features[:3]
    
    # update automatic dataset name
    new_dataset_name = 'new_dataset_' + str(len(table_data.ALL_DATASETS.keys()))

    # update disabled button
    disabled = len(list(table_data.ALL_DATASETS.keys())) < 2
    
    save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)
    
    return is_open, options_datasets, dataset_name, new_dataset_name, options_features, value_overview, options_features, value_parameter, disabled

@app.callback(
    Output("alert_transformation_table_data_duplicate_dataset", "is_open"),
    Input("button_transformation_table_data_add_dataset", "n_clicks"),
    State("input_transformation_table_data_new_dataset", "value"),
)
def toggle_add_dataset_alert(n_clicks, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    if dataset_name in list(table_data.ALL_DATASETS.keys()):
        return True    
    
    return False

@app.callback(
    Output("alert_transformation_table_data_duplicate_feature_name", "is_open"),
    Input("button_transformation_table_data_show", "n_clicks"),
    State("input_transformation_table_data_pca_feature_name", "value"),
    State("dropdown_transformation_table_data_dataset", "value"),
)
def toggle_feature_alert(n_clicks, feature_name, dataset_name):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    if feature_name in list(df.columns):
        return True
    
    return False

# update after feature removal
@app.callback(
    Output("dropdown_transformation_table_data_overview_feature", "options"),
    Output("dropdown_transformation_table_data_overview_feature", "value"),
    Output("dropdown_transformation_table_data_features", "options"),
    Output("dropdown_transformation_table_data_features", "value"),
    Output("dropdown_classification_target", "options"),
    Output("dropdown_classification_target", "value"),
    # input
    State('datatable_transformation_table_data_features', 'data_previous'),
    Input('datatable_transformation_table_data_features', 'data'),
    State("dropdown_transformation_table_data_dataset", "value"),
    State("dropdown_transformation_table_data_overview_feature", "value"),
    State("dropdown_transformation_table_data_features", "value"),
    State("dropdown_classification_dataset", "options"),
    State("dropdown_classification_dataset", "value"),
    State("dropdown_classification_target", "options"),
    State("dropdown_classification_target", "value"),
)
def delete_feature(previous, current, dataset_name, value_overview, value_parameter, options_dataset_name_supervised, value_dataset_name_supervised, options_target_supervised, value_target_supervised): 
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
    if dataset_name == value_dataset_name_supervised:
        options_target_supervised = options_features
        value_target_supervised = options_target_supervised[0]
        
    # update states
    save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)
        
    return options_features, value_overview, options_features, value_parameter, options_target_supervised, value_target_supervised

# update after selected dataset changes
@app.callback(
    Output('datatable_transformation_table_data_features', 'data'),
    Output('datatable_transformation_table_data_features', 'data_previous'),
    Output("dropdown_transformation_table_data_overview_feature", "options"),
    Output("dropdown_transformation_table_data_overview_feature", "value"),
    Output("rangeslider_transformation_table_data_overview", "value"),
    Output("dropdown_transformation_table_data_features", "options"),
    Output("dropdown_transformation_table_data_features", "value"),
    Output('checklist_transformation_table_data_all_features', 'value'),
    Input("dropdown_transformation_table_data_dataset", "value"),
)
def update_after_dataset_changes(dataset_name):
    if dataset_name is None or dataset_name == 'New Dataset...' or dataset_name == '':
        return dash.no_update
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    # update datatable
    data = pd.DataFrame({'Features': df.columns}).to_dict('records')
    
    # update overview line plot
    df = table_data.ALL_DATASETS[dataset_name]
    options_features = list(df.columns)
    value_overview = options_features[:4]
    range_values =  table_data.ALL_RANGES[dataset_name]
    
    # update parameter
    checklist_value = []
    value_parameter = options_features[:3]
    
    
    return data, None, options_features, value_overview, range_values, options_features, value_parameter, checklist_value

# update selected features
@app.callback(
    Output('dropdown_transformation_table_data_features', 'value'),
    Input("checklist_transformation_table_data_all_features", "value"),
    State('dropdown_transformation_table_data_features', 'options'),
)
def update_selected_features(all_features, options):
    if all_features is None or all_features == []:
        return dash.no_update
    
    return options

# update overview plot
@app.callback(
    Output("figure_transformation_table_data_overview", "figure"),
    Input("dropdown_transformation_table_data_overview_feature", "value"),
    Input("dropdown_transformation_table_data_plots", "value"),
    Input("dropdown_transformation_table_data_dataset", "value"),
    Input("rangeslider_transformation_table_data_overview", "value"),
)
def update_overview_plot(cols, plot, dataset_name, values_range):
    if cols is None or cols == "":
        return dash.no_update
    if dataset_name is None or dataset_name == 'New Dataset...' or dataset_name == '':
        return dash.no_update

    value_min = values_range[0]
    value_max = values_range[1]
    
    df = table_data.ALL_DATASETS[dataset_name].loc[value_min:value_max]
    if plot == PLOTS[0]:
        df = compute_plot(df, None, cols, value_min, value_max, reset_index=True)
        figure = get_overview_line_plot(df, cols)
    elif plot == PLOTS[1]:
        df = compute_plot(df, None, cols, value_min, value_max, reset_index=True)
        figure = get_overview_histogram_plot(df, cols)
    elif plot == PLOTS[2]:
        corr = analyse_correlation(df[cols])
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
    table_data.ALL_RANGES[dataset_name] = values_range
    save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)
    
    return figure

# update parameter
@app.callback(
    Output("container_transformation_table_data_pca", "style"),
    Input("dropdown_transformation_table_data_feature_transformation", "value"),
    State("container_transformation_table_data_pca", "style")
)
def update_style_pca(method, style):
    if style is None:
        style = {}
    if method == TRANSFORMATIONS_TABLE_DATA[2]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("slider_transformation_table_data_pca_n_components", "max"),
    Output("input_transformation_table_data_pca_feature_name", "value"),
    Input("dropdown_transformation_table_data_features", "value"),
)
def update_style_pca_components(value):
    if value is None or value == "":
        return dash.no_update
    
    max_n_components = len(value) - 1
    
    feature_name = '_'.join(value)
    
    return max_n_components, feature_name

@app.callback(
    Output("slider_transformation_table_data_dwt_vanishing_moments", "min"),
    Input("dropdown_transformation_table_data_dwt_wavelet", "value")
)
def update_style_dwt_n(wavelet):
    if wavelet == list(WAVELETS.keys())[1]:
        min_n = 2
    else:
        min_n = 1
    
    return min_n

@app.callback(
    Output("container_transformation_table_data_dwt", "style"),
    Input("dropdown_transformation_table_data_feature_transformation", "value"),
    State("container_transformation_table_data_dwt", "style")
)
def update_style_dwt(method, style):
    if style is None:
        style = {}
        
    if method == TRANSFORMATIONS_TABLE_DATA[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

@app.callback(
    Output("container_transformation_table_data_dwt_vanishing_moments", "style"),
    Input("dropdown_transformation_table_data_feature_transformation", "value"),
    Input("dropdown_transformation_table_data_dwt_wavelet", "value"),
    State("container_transformation_table_data_dwt_vanishing_moments", "style")
)
def update_style_dwt(method, wavelet, style):
    if style is None:
        style = {}
    if method == TRANSFORMATIONS_TABLE_DATA[3] and wavelet != list(WAVELETS.keys())[3]:
        style['display'] = 'block'
    else:
        style['display'] = 'none'        
    return style

  

# update after selected dataset changes
@app.callback(
    Output("dropdown_classification_dataset", "options"),
    Output("dropdown_classification_dataset", "value"),    
    Output("rangeslider_transformation_table_data_overview", "min"),
    Output("rangeslider_transformation_table_data_overview", "max"),    
    Output("rangeslider_transformation_table_data_overview", "value"),
    Input("dropdown_transformation_table_data_dataset", "options"),
    State("dropdown_classification_dataset", "value")
)
def update_after_dataset_changes(datasets, dataset_name):
    if datasets is None or datasets == []:
        return dash.no_update
    
    
    options = list(table_data.ALL_DATASETS.keys())
    if dataset_name not in options:
        dataset_name = options[0]
        
    df = table_data.ALL_DATASETS[dataset_name]
    
    # update rangeslider
    min_range = df.index.min() 
    max_range = df.index.max()
    value_range = table_data.ALL_RANGES[dataset_name]
    
    
    return options, dataset_name, min_range, max_range, value_range

# update button styles
@app.callback(
    Output("button_transformation_table_data_apply", "style"),
    Output("button_transformation_table_data_show", "style"),
    Input("button_transformation_table_data_show", "n_clicks"),
    Input("dropdown_transformation_table_data_dataset", "value"),
    Input("dropdown_transformation_table_data_features", "value"),
    Input("dropdown_transformation_table_data_feature_transformation", "value"),
    Input("slider_transformation_table_data_pca_n_components", "value"),
    Input("input_transformation_table_data_pca_feature_name", "value"),
    Input("dropdown_transformation_table_data_dwt_wavelet", "value"),
    Input("slider_transformation_table_data_dwt_mode", "value"),
    Input("slider_transformation_table_data_dwt_level", "value"),
    Input("slider_transformation_table_data_dwt_vanishing_moments", "value"),
    Input('datatable_transformation_table_data_features', 'data'),
    Input('datatable_transformation_table_data_features', 'data_previous'),
    State("button_transformation_table_data_apply", "style"),
    State("button_transformation_table_data_show", "style")
)
def update_style_buttons(n_clicks, dataset_name, v2, v3, v4, pca_feature_name, v6, v7, v8, v9, v10, v11, style_apply, style_show):   
    if style_apply is None:
        style_apply = {}
    if style_show is None:
        style_show = {}   
        
    if len(table_data.ALL_DATASETS.keys()) == 0:
        style_show['display'] = 'block'
        style_apply['display'] = 'none' 
        return style_apply, style_show
    
    if dataset_name == 'New Dataset...':
        return dash.no_update
        
    df = table_data.ALL_DATASETS[dataset_name]
    
    if pca_feature_name in list(df.columns):
        return dash.no_update
    
    triggered_id = ctx.triggered_id
    
    if triggered_id is None or triggered_id == 'button_transformation_table_data_show':
        style_apply['display'] = 'block'
        style_show['display'] = 'none'   
    else:
        style_show['display'] = 'block'
        style_apply['display'] = 'none' 
        
    return style_apply, style_show


# update line plot
@app.callback(
    # update preview plot
    Output("loading_transformation_table_data_preview", "children"),
    # inputs
    Input("button_transformation_table_data_show", "n_clicks"),
    State("dropdown_transformation_table_data_dataset", "value"),
    State("dropdown_transformation_table_data_features", "value"),
    State("dropdown_transformation_table_data_feature_transformation", "value"),
    State("slider_transformation_table_data_pca_n_components", "value"),
    State("input_transformation_table_data_pca_feature_name", "value"),
    State("dropdown_transformation_table_data_dwt_wavelet", "value"),
    State("slider_transformation_table_data_dwt_mode", "value"),
    State("slider_transformation_table_data_dwt_level", "value"),
    State("slider_transformation_table_data_dwt_vanishing_moments", "value"),
)
def update_preview_plot(n_clicks, dataset_name, cols, method, pca_n_components, pca_feature_name, dwt_wavelet, dwt_mode, dwt_level, dwt_vanishing_moments):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    
    df = table_data.ALL_DATASETS[dataset_name]
    
    if pca_feature_name in list(df.columns):
        return dash.no_update
    
    # read out parameter
    params = {}
    if method == TRANSFORMATIONS_TABLE_DATA[2]: # pca
        params['n_components'] = pca_n_components
        params['feature_name'] = pca_feature_name
    elif method == TRANSFORMATIONS_TABLE_DATA[3]: # dwt       
        params['mode'] = WAVELET_MODES[dwt_mode]
        params['level'] = dwt_level
        params['wavelet'] = WAVELETS[dwt_wavelet]
        if dwt_wavelet != list(WAVELETS.keys())[3]:
            params['n'] = dwt_vanishing_moments
    
    # apply transformation
    df, cols = apply_transformation_table_data(df, cols, method, params)

    # update line plot preview  
    figure = get_overview_line_plot(df, cols, index=None)
    graph = dcc.Graph(id="figure_transformation_table_data_preview", className='graph_categorical', figure=figure)
    
    return graph

# update df and overview
@app.callback(
    # update preview plot
    Output('datatable_transformation_table_data_features', 'data'),
    Output('datatable_transformation_table_data_features', 'data_previous'),
    # update parameter
    Output("dropdown_transformation_table_data_features", "options"),
    Output("dropdown_transformation_table_data_features", "value"),
    # update overview
    Output("dropdown_transformation_table_data_overview_feature", "options"),
    Output("dropdown_transformation_table_data_overview_feature", "value"),
    # inputs
    Input("button_transformation_table_data_apply", "n_clicks"),
    State("dropdown_transformation_table_data_dataset", "value"),
    State("dropdown_transformation_table_data_features", "value"),
    State("dropdown_transformation_table_data_feature_transformation", "value"),
    State("slider_transformation_table_data_pca_n_components", "value"),
    State("input_transformation_table_data_pca_feature_name", "value"),
    State("dropdown_transformation_table_data_dwt_wavelet", "value"),
    State("slider_transformation_table_data_dwt_mode", "value"),
    State("slider_transformation_table_data_dwt_level", "value"),
    State("slider_transformation_table_data_dwt_vanishing_moments", "value"),
    State("dropdown_transformation_table_data_overview_feature", "value"),
)
def update_overview(n_clicks, dataset_name, cols, method, pca_n_components, pca_feature_name, dwt_wavelet, dwt_mode, dwt_level, dwt_vanishing_moments, value_overview):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update
    
    # read out parameter
    params = {}
    if method == TRANSFORMATIONS_TABLE_DATA[2]: # pca
        params['n_components'] = pca_n_components
        params['feature_name'] = pca_feature_name
    elif method == TRANSFORMATIONS_TABLE_DATA[3]: # dwt       
        params['mode'] = WAVELET_MODES[dwt_mode]
        params['level'] = dwt_level
        params['wavelet'] = WAVELETS[dwt_wavelet]
        if dwt_wavelet != list(WAVELETS.keys())[3]:
            params['n'] = dwt_vanishing_moments
    
    # apply transformation
    table_data.ALL_DATASETS[dataset_name], _ = apply_transformation_table_data(table_data.ALL_DATASETS[dataset_name], cols, method, params)
    
    # update overview
    options_features = list(table_data.ALL_DATASETS[dataset_name].columns)
    if method == TRANSFORMATIONS_TABLE_DATA[2]:
        for col in cols:
            remove_item_if_exist(value_overview, col)        
    if len(value_overview) == 0:
        value_overview = [options_features[0]]
        
    # update paramter
    value_parameter = options_features[:3]
    
    # save changes
    save_dataset_states(table_data.ALL_DATASETS, table_data.ALL_RANGES)

    # update feature table
    data = pd.DataFrame({'Features': table_data.ALL_DATASETS[dataset_name].columns}).to_dict('records')
    
    return data, None, options_features, value_parameter, options_features, value_overview

