import dash_bootstrap_components as dbc
from dash import dcc
from dash import Input, Output, html, State, no_update, ctx
import dash
from dash import dash_table

# import app
from view.app import app

# import data
from data.table_data import *

# import supervised methods
from methods.data_analysis.supervised_learning import *

# import analyse methods
from methods.data_exploration.analyse import *

# import callbacks 
from callbacks.page_supervised_regression_callbacks import *

def create_data_supervised_panel():
    # create layout
    layout = html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Summary: Average Cross Validation Scores"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                            create_container_for_summary()
                        ),
                ],
                className='card_container'
            ),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Model Training"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_result_plot(),
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
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            "Prediction"
                        ],
                        className='card_header'
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    create_prediction_plot(),
                                    create_feature_importance_table(),
                                ],
                                align = "start",
                                justify = 'center',
                            ),
                        ],
                    ),
                ],
                className='card_container'
            ),
        ]
    )

    return layout

# summary 
def create_container_for_summary():    
    layout = html.Div(
        dbc.Col(
            dcc.Graph(
                id="analysis_regression_summary",
                className='graph_figure')
        ),
    )

    return layout

# current result
def create_result_plot():

    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading_regression_preview",
                        type="default",
                        children=dcc.Graph(id="figure_regression_result",
                              className='graph_categorical',
                             )
                    )
                    
                ]
            )
        ),
        
        width=8
    )
            

    return layout

# current result
def create_prediction_plot():

    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading_regression_prediction",
                        type="default",
                        children=dcc.Graph(id="figure_regression_prediction",
                              className='graph_categorical',
                             )
                    )
                    
                ]
            )
        ),
        
        width=8
    )
            

    return layout

def create_feature_importance_table():

    layout = dbc.Col( 
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading_regression_feature_importance",
                        type="default",
                        children=[
                            dash_table.DataTable(
                                id='datatable_regression_feature_importance',
                                columns= [{"name": "Feature", "id": "Feature"}, {"name": "Importance", "id": "Importance"}],
                                filter_action='native',
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
                        ]
                    )
                    
                ]
            )
        ),
        
        width=4
    )
            

    return layout

# parameter
def create_container_for_parameter():
    datasets =  list(table_data.ALL_DATASETS.keys())
    if len(datasets) > 0:
        init_dataset = datasets[0]
        targets = list(ALL_DATASETS[init_dataset].columns)
        init_target = targets[0]
    else:
        init_dataset = None
        targets = []
        init_target = None
    
    layout = dbc.Col( 
        dbc.Card([
            dbc.CardHeader("Parameter", className='card_header'),
            dbc.CardBody(
                [
                    dbc.Card([
                        dbc.CardHeader("Dataset:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_regression_dataset',
                                options=[{'label': i, 'value': i} for i in datasets],
                                value= init_dataset,
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ]),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Target:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_regression_target',
                                options=[{'label': i, 'value': i} for i in targets],
                                value= init_target,
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Train-Test-Split:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Slider(
                                min=0.1,
                                max=0.9,
                                step=0.1,
                                value=0.8,
                                id='slider_regression_train_test_split',
                            ),
                            
                            dcc.Link(
                                html.Img(src='/assets/img/link.png', id='img_regression_time_series_cv', className='tooltip_img'),
                                id='link_regression_time_series_cv',
                                href=TS_CROSS_VALIDATION_LINKS[0],
                                target='_blank',
                            ),
                            dbc.Tooltip(TS_CROSS_VALIDATION_DESCRIPTION[0], target='img_regression_time_series_cv', id='tooltip_regression_time_series_cv'),
                            
                            dcc.Checklist(
                                id='checklist_regression_time_series_crossvalidation',
                                options=['Use time series cross validation'],
                                value=[],
                                inputStyle={
                                    "margin-right": "0.5rem",
                                    "margin-top": "2rem",
                                },
                            ),
                            
                            
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Evaluation Scoring:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_regression_scoring',
                                options=[{'label': i, 'value': i} for i in REGRESSOR_SCORING.keys()],
                                value= list(REGRESSOR_SCORING.keys())[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader([
                            "Model:",
                            dcc.Link(
                                html.Img(src='/assets/img/link.png', id='img_regression_strategy', className='tooltip_img'),
                                id='link_regression_strategy',
                                href=REGRESSOR_LINKS[0],
                                target='_blank',
                            ),
                            dbc.Tooltip(REGRESSOR_DESCRIPTION[0], target='img_regression_strategy', id='tooltip_regression_strategy'),
                        ], className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_regression_model',
                                options=[{'label': i, 'value': i} for i in REGRESSOR],
                                value= REGRESSOR[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_baseline(),
                    create_container_for_random_forest(),
                    create_container_for_xgboost(),
                    
                    dbc.Card([
                        dbc.CardHeader("Model Name:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Input(
                                id='input_regression_model_name',
                                value='Baseline 1 Accuracy',
                                className='dropdown_overview_multi_feature',
                                required=True,
                                type='text',
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_alerts(),
                    
                    
                    html.Div([
                        dbc.Button("Save", color = "primary", id="button_regression_apply", className='btn_apply')
                    ], className="btn_aligned_right"),
                    
                    html.Div([
                        dbc.Button("Show", color = "primary", id="button_regression_show", className='btn_apply')
                    ], className="btn_aligned_right"),
                ],
                
            ),
            
        ],
        ),
        
        width=4
    )

    return layout

# baseline
def create_container_for_baseline():
    layout = html.Div(
        [
            add_container_for_strategy('container_regression_baseline_strategy', 'dropdown_regression_baseline_strategy'),
            add_container_for_quantile('container_regression_baseline_quantile', 'input_regression_baseline_quantile'),
            add_container_for_constant('container_regression_baseline_constant', 'input_regression_baseline_constant'),
            add_container_for_look_back('container_regression_baseline_look_back', 'slider_regression_baseline_look_back'),
        ],
        style={'display': 'block'},
        id='container_regression_baseline'
    )
    
    return layout

def add_container_for_strategy(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Strategy:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in REGRESSOR_BASELINE_STRATEGY.keys()],
                value= list(REGRESSOR_BASELINE_STRATEGY.keys())[0],
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

def add_container_for_constant(id_container, id_input):
    layout = dbc.Card([
        dbc.CardHeader("Constant:", className='card_subheader'),
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

def add_container_for_quantile(id_container, id_slider):
    layout = dbc.Card([
        html.Img(id='img_regression_baseline_quantile', src="/assets/img/tooltip.png", className='tooltip_img'),
            dbc.Tooltip("A quantile of 0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the maximum.", target='img_regression_baseline_quantile'),
        dbc.CardHeader("Quantile:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                min=0.0,
                max=1.0,
                step=0.05,
                value=0.5,
                marks={
                    0:"0.0",
                    0.1:"0.1",
                    0.2:"0.2",
                    0.3:"0.3",
                    0.4:"0.4",
                    0.5:"0.5",
                    0.6:"0.6",
                    0.7:"0.7",
                    0.8:"0.8",
                    0.9:"0.9",
                    1:"1.0",
                },
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'none'},
        id=id_container,
    )
        
    return layout

def add_container_for_look_back(id_container, id_slider):
    layout = dbc.Card([
        html.Img(id='img_regression_baseline_look_back', src="/assets/img/tooltip.png", className='tooltip_img'),
            dbc.Tooltip("Uses the mean of the previous values over the selected look back as prediction.", target='img_regression_baseline_look_back'),
        dbc.CardHeader("Look Back:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                min=1,
                max=20,
                step=1,
                value=5,
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
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ],
            
        ),
    ],
        className='card_subcontainer',
        style={'display': 'none'},
        id=id_container,
    )
        
    return layout

# random forest and xgboost
def create_container_for_random_forest():
    layout = html.Div(
        [
            add_container_for_n_estimators('container_regression_random_forest_n_estimators', 'slider_regression_random_forest_n_estimators'),
            add_container_for_criterion('container_regression_random_forest_criterion', 'slider_regression_random_forest_criterion'),
            add_container_for_max_depth('container_regression_random_forest_max_depth', 'slider_regression_random_forest_max_depth'),
            #add_container_for_warm_start('container_regression_random_forest_warm_start', 'slider_regression_random_forest_warm_start'),
        ],
        style={'display': 'none'},
        id='container_regression_random_forest'
    )
    
    return layout

def create_container_for_xgboost():
    layout = html.Div(
        [
            add_container_for_n_estimators('container_regression_xgboost_n_estimators', 'slider_regression_xgboost_n_estimators'),
            add_container_for_max_depth('container_regression_xgboost_max_depth', 'slider_regression_xgboost_max_depth'),
            add_container_for_learning_rate('container_regression_xgboost_learning_rate', 'slider_regression_xgboost_learning_rate'),
        ],
        style={'display': 'none'},
        id='container_regression_xgboost'
    )
    
    return layout

def add_container_for_n_estimators(id_container, id_slider):
    layout = dbc.Card([        
        dbc.CardHeader("Number of Estimators:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                step=10,
                value=50,
                marks={
                    10:"10",
                    50:"50",
                    100:"100",
                    150:"150",
                    200:"200",
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

def add_container_for_criterion(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Criterion:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in REGRESSOR_RF_CRITERION.keys()],
                value= list(REGRESSOR_RF_CRITERION.keys())[0],
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

def add_container_for_max_depth(id_container, id_slider):
    layout = dbc.Card([        
        dbc.CardHeader("Max Depth:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                step=1,
                value=36,
                marks={
                    1:"1",
                    5:"5",
                    10:"10",
                    15:"15",
                    20:"20",
                    25:"25",
                    30:"30",
                    36:"None",
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

def add_container_for_warm_start(id_container, id_checklist):
    layout = dbc.Card([
        html.Img(id='img_regression_random_forest_warm_start', src="/assets/img/tooltip.png", className='tooltip_img'),
            dbc.Tooltip("If checked, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.", target='img_regression_random_forest_warm_start'),
        dbc.CardHeader("Warm Start:", className='card_subheader'),
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

def add_container_for_learning_rate(id_container, id_slider):
    layout = dbc.Card([        
        dbc.CardHeader("Learning Rate:", className='card_subheader'),
        dbc.CardBody([
            dcc.Slider(
                id=id_slider,
                value=0.1,
                min=0.01,
                max=0.3,
                step=0.01,
                marks={
                    0.01:"0.01",
                    0.1:"0.1",
                    0.2:"0.2",
                    0.3:"0.3",
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

# alerts
def create_container_for_alerts():        
    layout = html.Div(
        [
             dbc.Alert("", id="alert_regression", is_open=False, color="danger", style={'display': 'block'}),
        ],
        style={'display': 'block'},
        id='container_regression_alerts'
    )
        
    return layout  


# # Define the page layout
layout = dbc.Container(
        [
            create_data_supervised_panel()
        ],
        id = "data_supervised_regression_container",
        fluid = True,
        style = {"display": "none"}
    )


