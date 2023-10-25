# Import necessary libraries
from dash import html, ctx, no_update
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import Input, Output, html,State
import numpy as np
import pandas as pd
from dash import dash_table

# import app
from view.app import app

# import data
from data import table_data

# import loading methods
from methods.data_exploration.loading import *

# import callbacks
from callbacks.page_data_loading_callbacks import *

def create_data_loading_panel():
    return html.Div([
        dbc.Card(
                [
                    dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Upload Data"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_container_drag_and_drop(),
                                    create_container_for_parameter(),
                                ],
                                align = "start",
                                justify = 'center',
                            ),
                        ],
                    ),
                ],
            ),
                ],
                id='container_loading',
                className='card_container'
            ),
        html.Div(id='output-data-upload')        
    ])


def create_container_drag_and_drop():
    layout = dbc.Col( 
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dcc.Upload(
                            id='upload_data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select File', className='upload_link'),
                            ]),
                            className='upload_data',
                            multiple=False,
                            style={
                                'textAlign': 'center'
                            }
                        )

                    ],
                    id='card_upload_data',
                    className='card_upload',
                ),
                dbc.Row(
                    [
                        html.Div(
                            [
                                html.P('File:', className='paragraph_loading', style={'display': 'inline-block', 'margin-right': '1rem'}),
                                html.P('', id='text_loading_selected_data', style={'display': 'inline-block'}),
                            ],
                            className='card_row',
                        )
                    ],
                    align="start",
                    justify='left',
                    className='card_row', 
                ),
            ],
            className='card_container'
        ),
        width=8
    )

    return layout

def create_container_for_parameter():
    layout = dbc.Col( 
        dbc.Card([
            dbc.CardHeader("Parameter", className='card_header'),
            dbc.CardBody(
                [
                    dbc.Card([
                        dbc.CardHeader("Datatype:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_loading_datatype',
                                options=[{'label': i, 'value': i} for i in DATA_TYPES],
                                value= DATA_TYPES[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    create_container_for_table_data_parameter(),
                    
                    dbc.Alert("There was an error loading the files. Please check the parameter.", id="alert_loading_files", is_open=False, color="danger", style={'display': 'block'}),
                    
                    html.Div([
                        dbc.Button("Load", color = "primary", id="button_load", className='btn_apply', disabled=True)
                    ], className="btn_aligned_right"),
                ],
                
            ),
            
        ],
        ),
        style={'display': 'none'},
        id='container_parameter_loading',
        width=4
    )
    
    return layout



def create_container_for_table_data_parameter():
    layout = html.Div(
        [
            add_container_for_seperator('container_loading_table_data_seperator', 'dropdown_loading_table_data_seperator'),
            add_container_for_index('container_loading_table_data_index', 'dropdown_loading_table_data_index'),
        ],
        style={'display': 'none'},
        id='container_loading_table_data_parameter'
    )
        
    return layout 

def add_container_for_index(id_container, id_dropdown):
    index_options = [{'label': 'None', 'value': 'none'}, {'label': 'Auto', 'value': 'auto'}]
    
    layout = dbc.Card([
        dbc.CardHeader([
            "Index:",
            html.Img(id='img_loading_index', src="/assets/img/tooltip.png", className='tooltip_img'),
            dbc.Tooltip(
                "The index is required for plotting. If your data has no index column or the index is a datetime, it is recommended to set this parameter to 'Auto' in order to add an extra index column.",
                target='img_loading_index', 
            ),
        ], className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=index_options,
                value=index_options[0]['value'],
                className='dropdown_overview_multi_feature',
                clearable=False,
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'none'},
        id=id_container,
    )
        
    return layout

def add_container_for_seperator(id_container, id_dropdown):
    seperator_options = list(SEPERATOR.keys())
    
    layout = dbc.Card([
        dbc.CardHeader("Seperator:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=seperator_options,
                value=seperator_options[0],
                className='dropdown_overview_multi_feature',
                clearable=False,
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'block'},
        id=id_container,
    )
        
    return layout


# Define the page layout
layout = dbc.Container(
        [
            create_data_loading_panel()
        ],
        id = "data_loading_container",
        fluid = True,
        style = {"display": "none"}
    )
