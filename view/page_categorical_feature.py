# Import necessary libraries
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash
import copy
import numpy as np
import pandas as pd
import json

from callbacks.page_categorical_feature_callbacks import *

# import data
from data import table_data

# import methods
from methods.data_exploration.analyse import get_num_numeric_categorical

# import figures
from view.page_helper_components.plots import get_numeric_categorical_ratio_plot, get_categorical_feature_pie_plot

import random



######################################
## First layer method: create panel ##
######################################

def create_data_categoricalfeature_panel():
    # create layout
    layout = html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Categorical Feature Ratio"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                            create_container_for_ratio_plot()
                        ),
                ],
                className='card_container'
            ),
            
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Feature Encoding"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_container_for_pie_plot(),
                                    create_container_for_parameter(),
                                ],
                                align = "start",
                                justify = 'center',
                            ),
                        ],
                    ),
                ],
                id='container_feature_encoding',
                className='card_container'
            ),

            
        ]
    )

    return layout


########################################
## second layer method: create detail ##
########################################
def create_container_for_ratio_plot():
    layout = html.Div(
        dbc.Col(
            dcc.Graph(
                id="exploration_categorical_feature_ratio_bar_plot",
                className='graph_figure')
        ),
    )

    return layout

def create_container_for_parameter():
    layout = dbc.Col( 
        dbc.Card([
            dbc.CardHeader("Parameter", className='card_header'),
            dbc.CardBody(
                [
                    dbc.Card([
                        dbc.CardHeader("Feature:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_categorical_feature',
                                options=[],
                                value=None,
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ]),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Method:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_categorical_strategy',
                                options=[{'label': i, 'value': i} for i in ENCODING_STRATEGIES],
                                value= ENCODING_STRATEGIES[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Replace:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_replace_value1',
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                            
                            html.P('with:'),
                            
                            dcc.Dropdown(
                                id='dropdown_replace_value2',
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],
                        
                        ),
                    ],
                        className='card_subcontainer',
                        id='card_categorical_replacement',
                        style = {"display": "none"}
                    ),
                    
                    create_container_for_alerts(),
                    
                    html.Div([
                        dbc.Button("Apply", color = "primary", id="button_categorical_apply", className='btn_apply')
                    ], className="btn_aligned_right")
                   
                ],
                
            ),
            
        ],
        ),
        
        width=4
    )

    return layout

def create_container_for_pie_plot():
    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(id="figure_categorical_feature_pie",
                              className='graph_categorical'
                             )
                ]
            )
        ),
        
        width=8
    )
            

    return layout

# alerts
def create_container_for_alerts():        
    layout = html.Div(
        [
             dbc.Alert("Column seems to contain unconvertable strings.", id="alert_categorical_unconvertable_string", is_open=False, color="danger", style={'display': 'block'}), 
        ],
        style={'display': 'block'},
        id='container_categorical_alerts'
    )
        
    return layout  

# # Define the page layout
layout = dbc.Container(
        [
            create_data_categoricalfeature_panel()
        ],
        id = "data_categorical_container",
        fluid = True,
        style = {"display": "none"}
    )
