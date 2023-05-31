# Import necessary libraries
from dash import html, no_update, MATCH, ALL, ctx
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_core_components as dcc
from dash import Input, Output, html, State, Dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# import app
from view.app import app

# import data
from data import table_data

# import callbacks
from callbacks.page_transformation_table_data_callbacks import *

# import analyse methods
from methods.data_exploration.analyse import *

# import transformation methods
from methods.data_transformation.transformations_table_data import *

# import state management
from states.states import *

# import plots
from view.page_helper_components.plots import *


def create_data_transformation_table_data_layout():
    # create layout
    layout = html.Div(
        [
            create_container_for_overview(),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Transformation"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_container_for_preview_plot(),
                                    create_container_for_parameter(),
                                ],
                                align = "start",
                                justify = 'center',
                            ),
                        ],
                    ),
                ],
                className='card_container'
            ),
            create_delete_dataset_modal(),
            create_add_dataset_modal(),
            
        ]
    )

    return layout

# overview
def create_container_for_overview():
    datasets = list(table_data.ALL_DATASETS.keys())
    if len(datasets) > 0:
        options_dataset =  datasets + ['New Dataset...']
        dataset_curr = options_dataset[0]

        df = table_data.ALL_DATASETS[dataset_curr]
        options_features = list(df.columns)

        # select index for slider
        min_value = df.index.min()
        max_value = df.index.max()

        value = table_data.ALL_RANGES[dataset_curr]
    else:
        options_dataset = []
        dataset_curr = None
        
        options_features = []
        
        min_value = 0
        max_value = 100
        
        value = [0, 100]

    # disabled 
    disabled = len(datasets) < 2
 
    
    layout = dbc.Card(
        [
            dbc.CardHeader(
                [
                    "Transformation Overview"
                ],
                className='card_header'
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            create_container_for_overview_plot(),
                            create_container_for_feature_table(),
                        ],
                        align = "start",
                        justify = 'center',
                    ),
                    dbc.Row(
                        [
                            dbc.Col([
                                dcc.RangeSlider(
                                    id="rangeslider_transformation_table_data_overview",
                                    min=min_value,
                                    max=max_value,
                                    tooltip={"placement": "top", "always_visible": False},
                                    value=value,
                                ),
                            ],
                                width=9
                            ),
                        ],
                        align = "start",
                    ),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='dropdown_transformation_table_data_overview_feature',
                                options=options_features,
                                value=options_features[:4],
                                className='dropdown_overview_multi_feature_full_width',
                                clearable=False,
                                multi=True,
                            ),
                        ],
                        width=7
                        ),
                         
                        dbc.Col([
                            dcc.Dropdown(
                                id='dropdown_transformation_table_data_plots',
                                options=PLOTS,
                                value=PLOTS[0],
                                className='dropdown_overview_single_feature_full_width',
                                clearable=False,
                            ),
                        ],
                        width=2
                        ),
                        
                        dbc.Col([
                        ],
                        width=1
                        ),
                        
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='dropdown_transformation_table_data_dataset',
                                        options=options_dataset,
                                        value=dataset_curr,
                                        className='dropdown_overview_single_feature_full_width',
                                        clearable=False,
                                    ),
                                    width=9
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        html.Img(src="/assets/img/delete.png", className='btn_img'),
                                        id='button_delete_dataset',
                                        disabled=disabled,
                                        color="#FDFCFC",
                                    ),
                                    className='btn_delete',
                                    width=1
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        html.Img(src="/assets/img/save.png", className='btn_img'),
                                        id='button_save_dataset',
                                        color="#FDFCFC",
                                    ),
                                    className='btn_delete',
                                    width=1
                                ),
                            ], justify='center', className='row_dropdown_and_button')
                        ], 
                            width=2)
                    ]),
                    
                ],
            ),
        ],
        id='container_transformation_table_data_overview',
        className='card_container'
    )
    
    return layout

def create_container_for_overview_plot():
    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(id="figure_transformation_table_data_overview",
                              className='graph_categorical',
                             )
                ]
            )
        ),
        
        width=9
    )
            
    return layout

def create_container_for_feature_table():
    datasets = list(table_data.ALL_DATASETS.keys())
    if len(datasets) > 0:
        options = list(table_data.ALL_DATASETS.keys()) + ['New Dataset...']
        value = options[0]
        data = pd.DataFrame({'Features': table_data.ALL_DATASETS[value].columns}).to_dict('records')
    else:
        options = []
        value = None
        data = pd.DataFrame({'Features': []}).to_dict('records')
    
    layout = dbc.Col(
        dbc.Card(
            dbc.CardBody([
                dash_table.DataTable(
                    id='datatable_transformation_table_data_features',
                    columns= [{"name": "Features", "id": "Features"}],
                    data=data,
                    filter_action='native',
                    row_deletable=True,
                    page_size=14,
                    fill_width=True,
                    sort_action="native",
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize' : "13pt"
                    },
                    fixed_rows={'headers': True},
                    style_cell={'textAlign': 'left', 'color': 'black'},
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'fontSize' : "13pt",
                        'minWidth': 50
                    },
                ),
            ]),
        ),
        width=3    
    )
    
    return layout


# line plot
def create_container_for_preview_plot():
    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading_transformation_table_data_preview",
                        type="default",
                        children=dcc.Graph(id="figure_transformation_table_data_preview",
                              className='graph_categorical',
                             )
                    )
                    
                ]
            )
        ),
        
        width=8
    )
            
    return layout

def create_container_for_parameter():
    datasets = list(table_data.ALL_DATASETS.keys())
    if len(datasets) > 0:
        options_dataset = list(table_data.ALL_DATASETS.keys()) + ['New Dataset...']
        dataset_curr = options_dataset[0]

        df = table_data.ALL_DATASETS[dataset_curr]
        options_features = list(df.columns)
    else:
        options_dataset = []
        dataset_curr = None

        options_features = []
    
    layout = dbc.Col( 
        dbc.Card([
            dbc.CardHeader("Parameter", className='card_header'),
            dbc.CardBody(
                [
                    dbc.Card([
                        dbc.CardHeader("Features:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_transformation_table_data_features',
                                options=options_features,
                                value=options_features[:3],
                                className='dropdown_overview_multi_feature',
                                clearable=True,
                                multi=True,
                            ),
                            dcc.Checklist(
                                id='checklist_transformation_table_data_all_features',
                                options=['Select all features'],
                                value=[],
                                inputStyle={
                                    "margin-right": "0.5rem",
                                },
                            )
                        ]),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Transformation:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_transformation_table_data_feature_transformation',
                                options=[{'label': i, 'value': i} for i in TRANSFORMATIONS_TABLE_DATA],
                                value=TRANSFORMATIONS_TABLE_DATA[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_pca_parameter(),
                    create_container_for_dwt_parameter(),
                    
                    create_container_for_alerts(),
                    
                    html.Div([
                        dbc.Button("Apply", color = "primary", id="button_transformation_table_data_apply", className='btn_apply')
                    ], className="btn_aligned_right"),
                    
                    html.Div([
                        dbc.Button("Show", color = "primary", id="button_transformation_table_data_show", className='btn_apply')
                    ], className="btn_aligned_right"),
                ],
                
            ),
            
        ],
        ),
        
        width=4
    )

    return layout

# modals (pop-ups)
def create_delete_dataset_modal():
    layout = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Delete Dataset?"), close_button=True),
                dbc.ModalBody("Are you sure to delete the dataset?"),
                dbc.ModalFooter([
                    dbc.Button(
                        "Yes",
                        id="button_transformation_table_data_delete_dataset_yes",
                        className="btn_apply",
                    ),
                    dbc.Button(
                        "No",
                        id="button_transformation_table_data_delete_dataset_no",
                        className="btn_apply",
                    ),
                ]),
            ],
            id="modal_transformation_table_data_delete_dataset",
            centered=True,
            is_open=False,
        )
    return layout

def create_add_dataset_modal():
    if len(table_data.ALL_DATASETS) > 0:
        index = str(len(table_data.ALL_DATASETS))
    else:
        index = "1"
    layout = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Add New Dataset"), close_button=True),
                dbc.ModalBody([
                    html.P("Name the new dataset:"),
                    dcc.Input(
                        id='input_transformation_table_data_new_dataset',
                        value='new_dataset_' + index,
                    ),
                dbc.Alert("Duplicate dataset name!", id="alert_transformation_table_data_duplicate_dataset", is_open=False, color="danger", style={'display': 'block'}),    
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Create",
                        id="button_transformation_table_data_add_dataset",
                        className="btn_apply",
                    ),
                ]),
            ],
            id="modal_transformation_table_data_add_dataset",
            centered=True,
            is_open=False,
            backdrop='static' 
        )
    return layout

# pca
def create_container_for_pca_parameter():        
    layout = html.Div(
        [
            add_container_for_n_components('container_transformation_table_data_pca_n_components', 'slider_transformation_table_data_pca_n_components'),
            add_container_for_feature_name('container_transformation_table_data_pca_feature_name', 'input_transformation_table_data_pca_feature_name')
        ],
        style={'display': 'block'},
        id='container_transformation_table_data_pca'
    )
        
    return layout   
    
def add_container_for_n_components(id_container, id_slider, min_compontents=1, value=2):
    layout = dbc.Card([
        dbc.CardHeader("Number of Components:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=min_compontents,
                 max=2,
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

def add_container_for_feature_name(id_container, id_input):
    layout = dbc.Card([
        dbc.CardHeader("New Feature Name:", className='card_subheader'),
        dbc.CardBody([
            dcc.Input(
                id=id_input,
                value='new_feature',
                className='dropdown_overview_multi_feature',
                required=True,
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'block'},
        id=id_container,
    )
        
    return layout

# dwt
def create_container_for_dwt_parameter():        
    layout = html.Div(
        [
            add_container_for_wavelet('container_transformation_table_data_dwt_wavelet', 'dropdown_transformation_table_data_dwt_wavelet'),
            add_container_for_mode('container_transformation_table_data_dwt_mode', 'slider_transformation_table_data_dwt_mode'),
            add_container_for_level('container_transformation_table_data_dwt_level', 'slider_transformation_table_data_dwt_level'),
            add_container_for_vanishing_moments('container_transformation_table_data_dwt_vanishing_moments', 'slider_transformation_table_data_dwt_vanishing_moments'),
        ],
        style={'display': 'block'},
        id='container_transformation_table_data_dwt'
    )
        
    return layout   
    
def add_container_for_wavelet(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Wavelet:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in WAVELETS.keys()],
                value= list(WAVELETS.keys())[0],
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

def add_container_for_mode(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Mode:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in WAVELET_MODES.keys()],
                value= list(WAVELET_MODES.keys())[0],
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

def add_container_for_level(id_container, id_slider):
    layout = dbc.Card([        
        dbc.CardHeader("Level:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                step=1,
                value=2,
                marks={
                    1:"1",
                    2:"2",
                    3:"3",
                    4:"4",
                    5:"5",
                    6:"None",
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

def add_container_for_vanishing_moments(id_container, id_slider):
    layout = dbc.Card([        
        dbc.CardHeader("Number of Vanishing Moments:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                step=1,
                value=1,
                marks={
                    1:"1",
                    2:"2",
                    3:"3",
                    4:"4",
                    5:"5",
                    6:"6",
                    7:"7",
                    8:"8",
                    9:"9",
                    10:"10",
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
    
# pca
def create_container_for_alerts():        
    layout = html.Div(
        [
            dbc.Alert("Duplicate feature name!", id="alert_transformation_table_data_duplicate_feature_name", is_open=False, color="danger", style={'display': 'block'}), 
        ],
        style={'display': 'block'},
        id='container_transformation_table_data_alerts'
    )
        
    return layout  

# Define the page layout
layout = dbc.Container(
        [
            create_data_transformation_table_data_layout()
        ],
        id = "data_transformation_table_data_container",
        fluid = True,
        style = {"display": "none"}
    )

