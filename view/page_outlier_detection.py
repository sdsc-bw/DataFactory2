# Import necessary libraries
from dash import html, no_update, MATCH, ALL, ctx
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import copy

# import app
from view.app import app

# import data
from data import table_data

# import callbacks
from callbacks.page_outlier_detection_callbacks import *

# import analyse methods
from methods.data_exploration.analyse import *

# import detection methods
from methods.data_exploration.outlier_detection import *

######################################
## First layer method: create panel ##
######################################

def create_data_outlier_detection_panel():
    # create layout
    layout = html.Div(
        [
            dbc.Row(
                [
                    create_violin_plot()
                    
                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'
            ),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Outlier Detection"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_container_outlier_plot(),
                                    create_container_for_parameter(),
                                ],
                                align = "start",
                                justify = 'center',
                            ),
                        ],
                    ),
                ],
                id='container_outlier_detection',
                className='card_container'
            ),
            dbc.Row(
                [
                    create_outlier_table()
                    
                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'
            ),
            
        ]
    )

    return layout


########################################
## second layer method: create detail ##
########################################
def create_violin_plot():
    
    layout = layout = dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Violin Distribution",
                        ],
                        className='card_header'),
                    dbc.CardBody(
                        [
                            dcc.Graph(id="figure_outlier_violin_plot", className='graph_figure'),
                        ]
                    ),
                ],
            ),
        ],
        className='card_container',
    )
    
    return layout

def create_container_outlier_plot():

    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading_outlier_preview",
                        type="default",
                        children=dcc.Graph(
                            id="figure_outlier_preview",
                            className='graph_categorical'
                        ),
                    )
                ]
            )
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
                        dbc.CardHeader("Method:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_outlier_method',
                                options=OUTLIER_DETECTION_METHODS,
                                value=OUTLIER_DETECTION_METHODS[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_random_forest_detector(),
                    create_container_for_density_detector(),
                    create_container_for_kv_detector(),
                    
                    html.Div([
                        dbc.Button("Remove", color = "primary", id="button_outlier_apply", className='btn_apply')
                    ], className="btn_aligned_right"),
                    
                    html.Div([
                        dbc.Button("Show", color = "primary", id="button_outlier_show", className='btn_apply')
                    ], className="btn_aligned_right"),
                ],
                
            ),
            
        ],
        ),
        
        width=4
    )

    return layout

# random forest detector
def create_container_for_random_forest_detector():
    layout = html.Div(
        [
            add_container_for_warm_start('container_outlier_random_forest_warm_start', 'check_outlier_random_forest_warm_start'),
            add_container_for_n_estimators('container_outlier_random_forest_n_estimators', 'slider_outlier_random_forest_n_estimators'),
        ],
        style={'display': 'block'},
        id='container_outlier_random_forest'
    )
    
    return layout
    
def add_container_for_warm_start(id_container, id_checklist):
    layout = dbc.Card([
        dbc.CardHeader([
            "Warm Start:",
            html.Img(id='img_outlier_warm_start', src="/assets/img/tooltip.png", className='tooltip_img'),
            dbc.Tooltip(
                "When selected, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.",
                target='img_outlier_warm_start', 
            ),
        ]
                       , className='card_subheader'),
        dbc.CardBody([
            dcc.Checklist(
                options=['Warm Start'],
                value=[],
                id=id_checklist,
                inputStyle={
                    "margin-right": "0.5rem",
                },
            )
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

def add_container_for_n_estimators(id_container, id_slider):
    layout = dbc.Card([        
        dbc.CardHeader("Number of Estimators:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                min=50,
                max=300,
                step=10,
                value=50,
                marks={
                    50:"50",
                    100:"100",
                    150:"150",
                    200:"200",
                    250:"250",
                    300:"300",
                },
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

# density detector
def create_container_for_density_detector():
    layout = html.Div(
        [
            add_container_for_n_neighbors('container_outlier_densitiy_n_neighbors', 'slider_outlier_densitiy_n_neighbors'),
#            add_container_for_contamination('container_outlier_densitiy_contamination', 'slider_outlier_densitiy_contamination')
            add_container_for_algorithm('container_outlier_densitiy_algorithm', 'dropdown_outlier_densitiy_algorithm'),
        ],
        style={'display': 'none'},
        id='container_outlier_densitiy'
    )
    
    return layout

def add_container_for_n_neighbors(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Number of Neighbors:", className='card_subheader'),
        dbc.CardBody([            
            dcc.Slider(
                id=id_slider,
                min=50,
                max=300,
                step=10,
                value=50,
                marks={
                    50:"50",
                    100:"100",
                    150:"150",
                    200:"200",
                    250:"250",
                    300:"300",
                },
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

def add_container_for_algorithm(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Algorithm:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in OUTLIER_DETECTION_LOCAL_ALGORITHM.keys()],
                value= list(OUTLIER_DETECTION_LOCAL_ALGORITHM.keys())[0],
                className='dropdown_overview_multi_feature',
                clearable=False,
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

def add_container_for_contamination(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Proportion of Outliers:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id="id_slider",
                min=0.01,
                max=0.99,
                step=0.01,
                value=0.1,
                marks={
                    0.01:"0.01",
                    0.1:"0.1",
                    0.2:"0.2",
                    0.3:"0.3",
                    0.4:"0.4",
                    0.5:"0.5",
                },
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

# kv detector
def create_container_for_kv_detector():
    layout = html.Div(
        [
            add_container_for_feature('container_outlier_kv_feature', 'dropdown_outlier_kv_feature'),
        ],
        style={'display': 'none'},
        id='container_outlier_kv'
    )
    
    return layout

def add_container_for_feature(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Feature:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[],
                value= None,
                className='dropdown_overview_multi_feature',
                clearable=False,
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

# outlier table
def create_outlier_table():

    layout = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader([
                    "Outlier Selection",
                    html.Img(id='img_outlier_selection', src="/assets/img/tooltip.png", className='tooltip_img'),
                    dbc.Tooltip(
                        "Here are the datapoints listed which are detected as outlier. You can manually deselect datapoints to keep them in the dataset.",
                        target='img_outlier_selection', 
                    ),
                ], className='card_header'),
                dbc.CardBody(
                    [
                    dash_table.DataTable(
                        id="table_outlier_detection",
                        columns=[],
                        editable=False,
                        row_selectable='multi',
                        page_size=20,
                        style_cell={
                            'height': 'auto',
                            # all three widths are needed
                            'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "Type"}, "textAlign": "left",}
                        ],
                        fill_width=True,
                        style_data={
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_table={'overflowX': 'auto'},
                    ),
                ],
                className = "twelve columns",
                style={"margin-top":"20px"}
            ),
        ],
        ),
        width = 12,
        className='card_container',
        id = 'card_outlier_datatable'        
    )

    return layout


# # Define the page layout
layout = dbc.Container(
        [
            create_data_outlier_detection_panel()
        ],
        id = "data_outlier_detection_container",
        fluid = True,
        style = {"display": "none"}
    )
