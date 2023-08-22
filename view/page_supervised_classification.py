import dash_bootstrap_components as dbc
from dash import dcc
from dash import Input, Output, html, State, no_update, ctx
import dash

# import app
from view.app import app

# import data
from data.table_data import *

# import supervised methods
from methods.data_analysis.supervised_learning import *

# import analyse methods
from methods.data_exploration.analyse import *

# import callbacks 
from callbacks.page_supervised_classification_callbacks import *

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
                            create_prediction_plot(),
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
                id="analysis_classification_summary",
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
                        id="loading_classification_preview",
                        type="default",
                        children=dcc.Graph(id="figure_classification_result",
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
                        id="loading_classification_prediction",
                        type="default",
                        children=dcc.Graph(id="figure_classification_prediction",
                              className='graph_categorical',
                             )
                    )
                    
                ]
            )
        ),
        
        width=12
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
                                id='dropdown_classification_dataset',
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
                                id='dropdown_classification_target',
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
                                id='slider_classification_train_test_split',
                            ),
                            
                            html.Img(id='img_classification_time_series_crossvalidation', src="/assets/img/tooltip.png", className='tooltip_img'),
                            dbc.Tooltip(
                                "If checked it keeps the order of the datapoints, otherwise it shuffles the dataset.",
                                target='img_classification_time_series_crossvalidation', 
                            ),
                            
                            dcc.Checklist(
                                id='checklist_classification_time_series_crossvalidation',
                                options=['Use time series cross validation'],
                                value=[],
                                inputStyle={
                                    "margin-right": "0.5rem",
                                    "margin-top": "2rem",
                                },
                            )
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Evaluation Scoring:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_classification_scoring',
                                options=[{'label': i, 'value': i} for i in CLASSIFIER_SCORING.keys()],
                                value= list(CLASSIFIER_SCORING.keys())[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    dbc.Card([
                        dbc.CardHeader("Model:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='dropdown_classification_model',
                                options=[{'label': i, 'value': i} for i in CLASSFIER],
                                value= CLASSFIER[0],
                                className='dropdown_overview_multi_feature',
                                clearable=False,
                            ),
                        ],),
                    ],
                        className='card_subcontainer',
                    ),
                    
                    create_container_for_baseline(),
                    create_container_for_knn(),
                    create_container_for_random_forest(),
                    create_container_for_xgboost(),
                    
                    dbc.Card([
                        dbc.CardHeader("Model Name:", className='card_subheader'),
                        dbc.CardBody([
                            dcc.Input(
                                id='input_classification_model_name',
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
                        dbc.Button("Save", color = "primary", id="button_classification_apply", className='btn_apply')
                    ], className="btn_aligned_right"),
                    
                    html.Div([
                        dbc.Button("Show", color = "primary", id="button_classification_show", className='btn_apply')
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
            add_container_for_strategy('container_classification_baseline_strategy', 'dropdown_classification_baseline_strategy'),
            add_container_for_constant('container_classification_baseline_constant', 'input_classification_baseline_constant'),
        ],
        style={'display': 'block'},
        id='container_classification_baseline'
    )
    
    return layout

def add_container_for_strategy(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Strategy:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in CLASSIFIER_BASELINE_STRATEGY.keys()],
                value= list(CLASSIFIER_BASELINE_STRATEGY.keys())[0],
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

# knn
def create_container_for_knn():
    layout = html.Div(
        [
            add_container_for_n_neighbors('container_classification_knn_n_neighbors', 'slider_classification_knn_n_neighbors'),
            add_container_for_algorithm('container_classification_knn_algorithm', 'dropdown_classification_knn_algorithm'),
            add_container_for_weights('container_classification_knn_weights', 'dropdown_classification_knn_weights'),
        ],
        style={'display': 'block'},
        id='container_classification_knn'
    )
    
    return layout

def add_container_for_n_neighbors(id_container, id_slider):
    layout = dbc.Card([
        dbc.CardHeader("Number of Neighbors:", className='card_subheader'),
        dbc.CardBody([            
            dcc.Slider(
                id=id_slider,
                min=1,
                max=100,
                step=1,
                value=5,
                marks={
                    1:"1",
                    10:"10",
                    20:"20",
                    30:"30",
                    40:"40",
                    50:"50",
                    60:"60",
                    70:"70",
                    80:"80",
                    90:"90",
                    100:"100",
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
                options=[{'label': i, 'value': i} for i in CLASSIFIER_KNN_ALGORITHM.keys()],
                value= list(CLASSIFIER_KNN_ALGORITHM.keys())[0],
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

def add_container_for_weights(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Weights:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in CLASSIFIER_KNN_WEIGHTS.keys()],
                value= list(CLASSIFIER_KNN_WEIGHTS.keys())[0],
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

# random forest and xgboost
def create_container_for_random_forest():
    layout = html.Div(
        [
            add_container_for_n_estimators('container_classification_random_forest_n_estimators', 'slider_classification_random_forest_n_estimators'),
            add_container_for_criterion('container_classification_random_forest_criterion', 'slider_classification_random_forest_criterion'),
            add_container_for_max_depth('container_classification_random_forest_max_depth', 'slider_classification_random_forest_max_depth'),
            #add_container_for_warm_start('container_classification_random_forest_warm_start', 'slider_classification_random_forest_warm_start'),
        ],
        style={'display': 'none'},
        id='container_classification_random_forest'
    )
    
    return layout

def create_container_for_xgboost():
    layout = html.Div(
        [
            add_container_for_n_estimators('container_classification_xgboost_n_estimators', 'slider_classification_xgboost_n_estimators'),
            add_container_for_max_depth('container_classification_xgboost_max_depth', 'slider_classification_xgboost_max_depth'),
            add_container_for_learning_rate('container_classification_xgboost_learning_rate', 'slider_classification_xgboost_learning_rate'),
        ],
        style={'display': 'none'},
        id='container_classification_xgboost'
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

def add_container_for_criterion(id_container, id_dropdown):
    layout = dbc.Card([
        dbc.CardHeader("Criterion:", className='card_subheader'),
        dbc.CardBody([
            dcc.Dropdown(
                id=id_dropdown,
                options=[{'label': i, 'value': i} for i in CLASSIFIER_RF_CRITERION.keys()],
                value= list(CLASSIFIER_RF_CRITERION.keys())[0],
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
             dbc.Alert("Invalid classes inferred from unique values of `y`. There might be missing classes or the wrong target selected.", id="alert_classification_missing_classes", is_open=False, color="danger", style={'display': 'block'}), 
            dbc.Alert("Number of cross validation splits cannot be greater than the number of members in each class. Please adapt train-test-split or the target.", id="alert_classification_invalid_splits", is_open=False, color="danger", style={'display': 'block'}),
            dbc.Alert("Number of neighbors is to large for number of Samples. Choose a smaller number of neigbors.", id="alert_classification_invalid_neighbors", is_open=False, color="danger", style={'display': 'block'}),
            dbc.Alert("", id="alert_classification", is_open=False, color="danger", style={'display': 'block'}),
        ],
        style={'display': 'block'},
        id='container_classification_alerts'
    )
        
    return layout  


# # Define the page layout
layout = dbc.Container(
        [
            create_data_supervised_panel()
        ],
        id = "data_supervised_classification_container",
        fluid = True,
        style = {"display": "none"}
    )


