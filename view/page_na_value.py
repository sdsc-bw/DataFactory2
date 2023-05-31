# Import necessary libraries
from dash import html, no_update, MATCH, ALL, ctx
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash
import plotly.express as px
import plotly.graph_objs as go
import copy
import numpy as np

# import app
from view.app import app

# import data
from data.table_data import *

# import filling methods
from methods.data_exploration.imputer import *

# import analyse methods
from methods.data_exploration.analyse import *

# import callbacks 
from callbacks.page_na_value_callbacks import *


def create_data_navalue_panel():
    # create layout
    layout = html.Div(
        [
            dbc.Row([
                create_na_bar_plot(),
                create_na_heatmap_plot(),
            ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'
            ),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "NA Imputing"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_container_line_plot(),
                                    create_container_for_parameter(),
                                ],
                                align = "start",
                                justify = 'center',
                            ),
                        ],
                    ),
                ],
                id='container_na_imputer',
                className='card_container'
            ),
        ]
    )

    return layout



########################################
## second layer method: create detail ##
########################################

def create_na_bar_plot():
    layout = dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Number of NA Values",
                        ],
                        className='card_header'),
                    dbc.CardBody(
                        [
                            dcc.Graph(id = "figure_na_bar_plot", className='graph_figure'),
                        ]
                    ),
                ],
            ),
        ],
        width = 6,
        className='card_container',
    )

    return layout

def create_na_heatmap_plot():
    
    layout = dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Position of NA Values",
                        ],
                        className='card_header'),
                    dbc.CardBody(
                        [
                            dcc.Graph(id = "figure_na_heatmap", className='graph_figure')
                        ]
                    ),
                ],
            ),
        ], 
        width = 6,
        className='card_container',
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
                                id='dropdown_na_feature',
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
                                id='dropdown_na_method',
                                options=[{'label': i, 'value': i} for i in IMPUTER_METHODS],
                                value= IMPUTER_METHODS[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_simple_parameter(),
                    create_container_for_iterative_parameter(),
                    create_container_for_knn_parameter(),
                    create_container_for_manual_parameter(),
                    
                    html.Div([
                        dbc.Button("Apply", color = "primary", id="button_na_apply", className='btn_apply')
                    ], className="btn_aligned_right"),
                    
                    html.Div([
                        dbc.Button("Show", color = "primary", id="button_na_show", className='btn_apply')
                    ], className="btn_aligned_right"),
                ],
                
            ),
            
        ],
        ),
        
        width=4
    )

    return layout


# simple imputer
def create_container_for_simple_parameter():
    
    layout = html.Div(
        [
            add_container_for_strategy('container_na_simple_strategy', 'dropdown_na_simple_strategy'),
            add_container_for_fill_value('container_na_simple_fill_value', 'input_na_simple_fill_value')
        ],
        style={'display': 'block'},
        id='container_na_simple'
    )
        
    return layout   

# iterative imputer
def create_container_for_iterative_parameter():
    
    layout = html.Div(
        [
            add_container_for_max_iter('container_na_iterative_max_iter', 'slider_na_iterative_max_iter'),
            add_container_for_n_nearest_features('container_na_iterative_n_nearest_features', 'slider_na_iterative_n_nearest_features'),
            add_container_for_initial_strategy('container_na_iterative_initial_strategy', 'dropdown_na_iterative_initial_strategy'),
            add_container_for_initial_fill_value('container_na_iterative_fill_value', 'dropdown_na_iterative_fill_value'),
            add_container_for_imputation_order('container_na_iterative_imputation_order', 'dropdown_na_iterative_imputation_order')
        ],
        style={'display': 'none'},
        id='container_na_iterative'
    )
        
    return layout  

# knn imputer
def create_container_for_knn_parameter():
    
    layout = html.Div(
        [
            add_container_for_n_neighbors('container_na_knn_n_neighbors', 'slider_na_knn_n_neighbors'),
            add_container_for_weights('container_na_iterative_weights', 'dropdown_na_iterative_weights')
        ],
        style={'display': 'none'},
        id='container_na_knn'
    )
        
    return layout  

# manual
def create_container_for_manual_parameter():
    
    layout = html.Div(
        [
            add_container_for_index('container_na_manual_index', 'input_na_manual_index'),
            add_container_for_fill_value('container_na_manual_fill_value', 'input_na_manual_fill_value')
        ],
        style={'display': 'none'},
        id='container_na_manual'
    )
        
    return layout 

# helper functions
def add_container_for_strategy(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Strategy:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in IMPUTER_STRATEGIES.keys()],
                value= list(IMPUTER_STRATEGIES.keys())[0],
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

def add_container_for_fill_value(id_container, id_input):
    layout = dbc.Card([
        dbc.CardHeader("Fill Value:", className='card_subheader'),
        dbc.CardBody([
            dcc.Input(
                id=id_input,
                value= 0,
                className='dropdown_overview_multi_feature',
                required=True,
                type='number',
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'block'},
        id=id_container,
    )
        
    return layout

def add_container_for_max_iter(id_container, id_slider, min_iter=1, max_iter=20, value=10):
    layout = dbc.Card([
        dbc.CardHeader("Max Iterations:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=min_iter, 
                 max=max_iter, 
                 marks = {i: {'label': str(round(i))} for i in np.arange(min_iter, max_iter, (max_iter-min_iter)/5)},
                 step=1,
                 value=value,
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout    

def add_container_for_n_nearest_features(id_container, id_slider, min_features=1, max_features=2):
    value = max_features
    layout = dbc.Card([
        dbc.CardHeader("Number of Nearest Features", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=min_features, 
                 max=max_features, 
                 marks = {i: {'label': str(round(i))} for i in np.arange(min_features, max_features, (max_features-min_features)/5)},
                 step=1,
                 value=value,
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout  

def add_container_for_initial_strategy(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Initial Strategy:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in IMPUTER_STRATEGIES.keys()],
                value= list(IMPUTER_STRATEGIES.keys())[0],
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

def add_container_for_initial_fill_value(id_container, id_input):
    layout = dbc.Card([
        dbc.CardHeader("Initial Fill Value:", className='card_subheader'),
        dbc.CardBody([
            dcc.Input(
                id=id_input,
                value= 0,
                className='dropdown_overview_multi_feature',
                required=True,
                type='number',
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'none'},
        id=id_container,
    )
        
    return layout

def add_container_for_imputation_order(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Imputation Order:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in IMPUTER_ORDER.keys()],
                value= list(IMPUTER_ORDER.keys())[0],
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

def add_container_for_n_neighbors(id_container, id_slider, min_neighbors=1, max_neighbors=10, value=5):
    layout = dbc.Card([
        dbc.CardHeader("Number of Neighbors:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=min_neighbors, 
                 max=max_neighbors, 
                 marks = {i: {'label': str(round(i))} for i in np.arange(min_neighbors, max_neighbors, (max_neighbors-min_neighbors)/5)},
                 step=1,
                 value=value,
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout 

def add_container_for_weights(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Imputation Order:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in IMPUTER_WEIGHTS.keys()],
                value= list(IMPUTER_WEIGHTS.keys())[0],
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

def add_container_for_index(id_container, id_input):
    layout = dbc.Card([
        dbc.CardHeader("Index:", className='card_subheader'),
        dbc.CardBody([
            dcc.Input(
                id=id_input,
                min=0,
                max=1,
                value= 0,
                className='dropdown_overview_multi_feature',
                required=True,
                type='number',
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'block'},
        id=id_container,
    )
        
    return layout


# line plot
def create_container_line_plot():

    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading_na_imputer_preview",
                        type="default",
                        children=dcc.Graph(
                            id="figure_na_imputer_preview",
                            className='graph_categorical'
                        ),
                    )
                ]
            )
        ),
        
        width=8
    )
            

    return layout

#####################
##    framework    ##
#####################

# Define the page layout
layout = dbc.Container(
        [
            create_data_navalue_panel()
        ],
        id = "data_na_value_container",
        fluid = True,
        style = {"display": "none"}
    )
