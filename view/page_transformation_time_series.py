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
from callbacks.page_transformation_time_series_callbacks import *

# import analyse methods
from methods.data_exploration.analyse import *

# import transformation methods
from methods.data_transformation.transformations_table_data import *

# import state management
from states.states import *

# import plots
from view.page_helper_components.plots import *

# import slider marks
from view.page_helper_components.sliders import get_slider_marks

def create_data_transformation_time_series_layout():
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
 
    marks = get_slider_marks((min_value, max_value))
    
    layout = dbc.Card(
        [
            dbc.CardHeader(
                [
                    "Transformation Overview", 
                    html.Img(id='img_transformation_time_series_overview', src="/assets/img/tooltip.png", className='tooltip_img'),
                    dbc.Tooltip(
                        "You need to create a dataset in order to train models. Add a new dataset with '+'. With the slider you can cut out an interval of the dataset that is later used for the analysis. You can create multiple datasets with different features and transformations.",
                        target='img_transformation_time_series_overview', 
                    ),
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
                                    id="rangeslider_transformation_time_series_overview",
                                    min=min_value,
                                    max=max_value,
                                    tooltip={"placement": "top", "always_visible": False},
                                    marks=marks,
                                    value=value,
                                    step=1,
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
                                id='dropdown_transformation_time_series_overview_feature',
                                options=options_features,
                                value=options_features[:4],
                                className='dropdown_overview_multi_feature_full_width',
                                clearable=False,
                                multi=True,
                            ),
                        ],
                        width=6
                        ),
                         
                        dbc.Col([
                            dcc.Dropdown(
                                id='dropdown_transformation_time_series_plots',
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
                                        id='dropdown_transformation_time_series_dataset',
                                        options=options_dataset,
                                        value=dataset_curr,
                                        className='dropdown_overview_single_feature_full_width',
                                        clearable=False,
                                    ),
                                    width=8
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        html.Img(src="/assets/img/plus.png", className='btn_img'),
                                        id='button_transformation_time_series_plus_dataset',
                                        color="#FDFCFC",
                                    ),
                                    className='btn_delete',
                                    width=1
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        html.Img(src="/assets/img/save.png", className='btn_img'),
                                        id='button_transformation_time_series_save_dataset',
                                        color="#FDFCFC",
                                    ),
                                    className='btn_delete',
                                    width=1
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        html.Img(src="/assets/img/delete.png", className='btn_img'),
                                        id='button_transformation_time_series_delete_dataset',
                                        disabled=disabled,
                                        color="#FDFCFC",
                                    ),
                                    className='btn_delete',
                                    width=1
                                ),
                                
                            ], justify='center', className='row_dropdown_and_button')
                        ], 
                            width=3)
                    ]),
                    
                ],
            ),
        ],
        id='container_transformation_time_series_overview',
        className='card_container'
    )
    
    return layout

def create_container_for_overview_plot():
    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(id="figure_transformation_time_series_overview",
                              className='graph_categorical',
                              #figure = get_categorical_feature_pie_plot(counts)
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
        options = list(table_data.ALL_DATASETS.keys())
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
                    id='datatable_transformation_time_series_features',
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
                        id="loading_transformation_time_series_preview",
                        type="default",
                        children=dcc.Graph(id="figure_transformation_time_series_preview",
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
        options_dataset = list(table_data.ALL_DATASETS.keys())
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
                                id='dropdown_transformation_time_series_features',
                                options=options_features,
                                value=options_features[:3],
                                className='dropdown_overview_multi_feature',
                                clearable=True,
                                multi=True,
                            ),
                            dcc.Checklist(
                                id='checklist_transformation_time_series_all_features',
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
                        dbc.CardHeader([
                            "Transformation:",
                            dcc.Link(
                                html.Img(src='/assets/img/tooltip.png', id='img_time_series_strategy', className='tooltip_img'),
                                id='link_time_series_strategy',
                                href='',
                                target='_blank',
                            ),
                            dbc.Tooltip(TRANSFORMATIONS_DESCRIPTIONS[0], target='img_time_series_strategy', id='tooltip_time_series_strategy'),
                        ], className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_transformation_time_series_feature_transformation',
                                options=[{'label': i, 'value': i} for i in TRANSFORMATIONS_TS],
                                value=TRANSFORMATIONS_TS[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_pca_parameter(),
                    create_container_for_dwt_parameter(),
                    create_container_for_shifting_parameter(),
                    create_container_for_sliding_window_parameter(),
                    create_container_for_differencing_parameter(),
                    create_container_for_savitzky_golay_filter_parameter(),
                    
                    create_container_for_alerts(),
                    
                    html.Div([
                        dbc.Button("Apply", color = "primary", id="button_transformation_time_series_apply", className='btn_apply')
                    ], className="btn_aligned_right"),
                    
                    html.Div([
                        dbc.Button("Show", color = "primary", id="button_transformation_time_series_show", className='btn_apply')
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
                        id="button_transformation_time_series_delete_dataset_yes",
                        className="btn_apply",
                    ),
                    dbc.Button(
                        "No",
                        id="button_transformation_time_series_delete_dataset_no",
                        className="btn_apply",
                    ),
                ]),
            ],
            id="modal_transformation_time_series_delete_dataset",
            centered=True,
            is_open=False,
        )
    return layout

def create_add_dataset_modal():
    layout = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Add New Dataset"), close_button=True),
                dbc.ModalBody([
                    html.P("Name the new dataset:"),
                    dcc.Input(
                        id='input_transformation_time_series_new_dataset',
                        value='new_dataset',
                    ),
                dbc.Alert("Duplicate dataset name!", id="alert_transformation_time_series_duplicate_dataset", is_open=False, color="danger", style={'display': 'block'}),    
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Create",
                        id="button_transformation_time_series_add_dataset",
                        className="btn_apply",
                    ),
                ]),
            ],
            id="modal_transformation_time_series_add_dataset",
            centered=True,
            is_open=False,
            backdrop='static' 
        )
    return layout

# pca
def create_container_for_pca_parameter():        
    layout = html.Div(
        [
            add_container_for_n_components('container_transformation_time_series_pca_n_components', 'slider_transformation_time_series_pca_n_components'),
            add_container_for_feature_name('container_transformation_time_series_pca_feature_name', 'input_transformation_time_series_pca_feature_name')
        ],
        style={'display': 'none'},
        id='container_transformation_time_series_pca'
    )
        
    return layout   
    
def add_container_for_n_components(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Number of Components:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=1,
                 max=2,
                 step=1,
                 value=1,
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
        id=id_container,
    )
        
    return layout

# dwt
def create_container_for_dwt_parameter():        
    layout = html.Div(
        [
            add_container_for_wavelet('container_transformation_time_series_dwt_wavelet', 'dropdown_transformation_time_series_dwt_wavelet'),
            add_container_for_mode('container_transformation_time_series_dwt_mode', 'slider_transformation_time_series_dwt_mode'),
            add_container_for_level('container_transformation_time_series_dwt_level', 'slider_transformation_time_series_dwt_level'),
            add_container_for_vanishing_moments('container_transformation_time_series_dwt_vanishing_moments', 'slider_transformation_time_series_dwt_vanishing_moments'),
            add_container_for_feature_name('container_transformation_time_series_dwt_feature_name', 'input_transformation_time_series_dwt_feature_name')
        ],
        style={'display': 'none'},
        id='container_transformation_time_series_dwt'
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

# shifting
def create_container_for_shifting_parameter():        
    layout = html.Div(
        [
            add_container_for_steps('container_transformation_time_series_shift_steps', 'slider_transformation_time_series_shift_steps'),
            add_container_for_multi_shift('container_transformation_time_series_shift_multi', 'checklist_transformation_time_series_shift_multi')
        ],
        style={'display': 'none'},
        id='container_transformation_time_series_shift'
    )
        
    return layout   

def add_container_for_steps(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Look Backs:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=1,
                 max=20,
                 step=1,
                 value=1,
                 marks={
                    1:"1",
                    3:"3",
                    5:"5",
                    7:"7",
                    9:"9",
                    11:"11",
                    13:"13",
                    15:"15",
                    17:"17",
                    19:"19",
                 },
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout 

def add_container_for_multi_shift(id_container, id_checklist):
    layout = dbc.Card([
        dbc.CardBody([
             dcc.Checklist(
                 id=id_checklist,
                 options=['Enable multi shift'],
                 value=[],
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

# sliding window
def create_container_for_sliding_window_parameter():        
    layout = html.Div(
        [
            add_container_for_operations('container_transformation_time_series_sw_operations', 'dropdown_transformation_time_series_sw_operations'),
            add_container_for_periods('container_transformation_time_series_sw_periods', 'slider_transformation_time_series_sw_periods'),
            
        ],
        style={'display': 'none'},
        id='container_transformation_time_series_sw'
    )
        
    return layout   

def add_container_for_operations(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Operation:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in SLIDING_WINDOW_OPERATIONS.keys()],
                value= list(SLIDING_WINDOW_OPERATIONS.keys())[0],
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

def add_container_for_periods(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Window Size:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=1,
                 max=20,
                 step=1,
                 value=2,
                 marks={
                    1:"1",
                    3:"3",
                    5:"5",
                    7:"7",
                    9:"9",
                    11:"11",
                    13:"13",
                    15:"15",
                    17:"17",
                    19:"19",
                 },
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout 

# differencing
def create_container_for_differencing_parameter():        
    layout = html.Div(
        [
            add_container_for_diff_periods('container_transformation_time_series_diff_periods', 'slider_transformation_time_series_diff_periods'),
        ],
        style={'display': 'none'},
        id='container_transformation_time_series_diff'
    )
        
    return layout   

def add_container_for_diff_periods(id_container, id_slider):
    layout = dbc.Card([
        
        dbc.CardHeader([
            "Periods:",
            html.Img(id='img_transformation_time_series_diff_periods', src="/assets/img/tooltip.png", className='tooltip_img'),
            dbc.Tooltip("Calculates the difference of a datapoint compared with another datapoint. A positive period compares it with the previous datapoint, a negative with the following datapoint. E.g. Period = 3 computes the difference with 3rd previous datapoint.", target='img_transformation_time_series_diff_periods'),
        ]
            , className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=-10,
                 max=10,
                 step=1,
                 value=1,
                 marks={
                    -10:"-10",
                    -8:"-8",
                    -6:"-6",
                    -4:"-4",
                    -2:"-2",
                    0:"0",
                    2:"2",
                    4:"4",
                    6:"6",
                    8:"8",
                    10:"10",
                 },
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout 

# savitzky golay filter
def create_container_for_savitzky_golay_filter_parameter():        
    layout = html.Div(
        [
            add_container_for_polyorder('container_transformation_time_series_sgf_polyorder', 'slider_transformation_time_series_sgf_polyorder'),
            add_container_for_sfg_periods('container_transformation_time_series_sgf_periods', 'slider_transformation_time_series_sgf_periods'),
            
        ],
        style={'display': 'none'},
        id='container_transformation_time_series_sgf'
    )
        
    return layout   

def add_container_for_polyorder(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Order of Polynomial:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                 id=id_slider, 
                 min=1,
                 max=20,
                 step=1,
                 value=1,
                 marks={
                    1:"1",
                    3:"3",
                    5:"5",
                    7:"7",
                    9:"9",
                    11:"11",
                    13:"13",
                    15:"15",
                    17:"17",
                    19:"19",
                 },
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout

def add_container_for_sfg_periods(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Window Size:", className='card_subheader'),
        dbc.CardBody([
             dcc.Slider(
                 id=id_slider, 
                 min=3,
                 max=21,
                 step=2,
                 value=1,
                 tooltip={"placement": "top", "always_visible": False},
             ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        id=id_container,
    )
        
    return layout 

# alerts
def create_container_for_alerts():        
    layout = html.Div(
        [
            dbc.Alert("Duplicate feature name!", id="alert_transformation_time_series_duplicate_feature_name", is_open=False, color="danger", style={'display': 'block'}), 
             dbc.Alert("The order of the polynomial must be less than the window size", id="alert_transformation_time_series_polyorder", is_open=False, color="danger", style={'display': 'block'}), 
        ],
        style={'display': 'block'},
        id='container_transformation_time_series_alerts'
    )
        
    return layout  

# Define the page layout
layout = dbc.Container(
        [
            create_data_transformation_time_series_layout()
        ],
        id = "data_transformation_time_series_container",
        fluid = True,
        style = {"display": "none"}
    )


