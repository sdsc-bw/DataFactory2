# Import necessary libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash, no_update, ctx
import plotly.express as px
import numpy as np

# import loading methods
from methods.data_exploration.loading import *

# import analysing methods
from methods.data_exploration.analyse import *

# import plots
from view.page_helper_components.plots import *

# import states
# import analysing methods
from states.states import *

# import app
from view.app import app

# import data
from data import table_data

@app.callback(Output('text_loading_selected_data', 'children'),
              Output('button_load', 'disabled'),
              Output('container_loading_table_data_parameter', 'style'),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'),
              Input('dropdown_loading_datatype', 'value'),
              State('container_loading_table_data_parameter', 'style'),
)
def update_parameter(contents, filename, datatype, style_table_data):
    if style_table_data is None:
        style_table_data = {}
    
    if contents is not None:
        if datatype == DATA_TYPES[0] or datatype == DATA_TYPES[1]:
            style_table_data['display'] = 'block'
        else: 
            style_table_data['display'] = 'none'
        disabled = False
    else:
        filename = ""
        style_table_data['display'] = 'none'
        disabled = True
    return filename, disabled, style_table_data 

@app.callback(Output('button_load', 'disabled'),
              Output('alert_loading_files', 'is_open'),
              Output("datatable_overview", "data"),
              Output("button_overview", 'disabled'),
              Output("button_categorical", 'disabled'),
              Output("button_na_values", 'disabled'),
              Output("button_outlier", 'disabled'),
              Output("button_ts", 'disabled'),
              Output("button_sc", 'disabled'),
              Output("button_sr", 'disabled'),
              Output("button_usl", 'disabled'),
              Input('button_load', 'n_clicks'),
              State('upload_data', 'contents'),
              State('upload_data', 'filename'),
              State('dropdown_loading_datatype', 'value'),
              State('dropdown_loading_table_data_seperator', 'value'),
              State('dropdown_loading_table_data_index', 'value'),
)
def load_data(n_clicks, contents, filename, datatype, sep, index):
    is_open = False
    data_datatable_overview = []
    columns_datatable_overview = []
    params = {}
    
    if n_clicks is None or n_clicks == 0:
        if table_data.DF_RAW is None:
            return 3 * [dash.no_update] + 8 * [True]
        else:
            cat_cols = table_data.DF_RAW.select_dtypes(include='object').columns
            nan_cols = table_data.DF_RAW.columns[table_data.DF_RAW.isna().any()].tolist()

            overview_disabled = False
            categorical_disabled = len(cat_cols) == 0
            na_disabled = len(nan_cols) == 0 or len(cat_cols) > 0
            rest_disabled = len(nan_cols) > 0 != [] or len(cat_cols) > 0
            return 3 * [dash.no_update] + [overview_disabled, categorical_disabled, na_disabled, rest_disabled, rest_disabled, rest_disabled, rest_disabled, rest_disabled]
    
    is_open = False
    if contents is not None: 
        if datatype == DATA_TYPES[0]:
            params['sep'] = SEPERATOR[sep]
            params['index'] = index
            df = parse_table_data(contents, filename, params)
            if df is not None:
                table_data.SEP = SEPERATOR[sep]
                table_data.DF_RAW = df
                # update overview datatable to trigger all other updates
                df_tmp = analyse_df(table_data.DF_RAW)
                data_datatable_overview = df_tmp.to_dict('records')
                # reset states
                table_data.ALL_RANGES = {}
                table_data.ALL_DATASETS = {}
                delete_dataset(IN_PROCESSING_DATASETNAME)
                delete_dataset_states()
            else:
                return [False] + [True] + 9 * [dash.no_update]
         # TODO add more datatypes
        
    cat_cols = table_data.DF_RAW.select_dtypes(include='object').columns
    nan_cols = table_data.DF_RAW.columns[table_data.DF_RAW.isna().any()].tolist()
        
    overview_disabled = False
    categorical_disabled = len(cat_cols) == 0
    na_disabled = len(nan_cols) == 0 or len(cat_cols) > 0
    rest_disabled = len(nan_cols) > 0 != [] or len(cat_cols) > 0
        
    training_disabled = True    
        
    return False, False, data_datatable_overview, overview_disabled, categorical_disabled, na_disabled, rest_disabled, rest_disabled, training_disabled, training_disabled, training_disabled

