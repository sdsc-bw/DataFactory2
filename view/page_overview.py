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

# import callbacks
from callbacks.page_overview_callbacks import *

# import analyse methods
from methods.data_exploration.analyse import *

# import figures
from view.page_helper_components.plots import get_numeric_categorical_ratio_plot, get_categorical_feature_pie_plot

# Define the page layout
def create_data_overview_panel():
    return html.Div(
        [
            dbc.Row(
                [
                    generate_text_board('text_board_shape', '(#Rows, #Features)', '(0,0)'),
                    generate_text_board('text_board_memory', 'Memory used', '0MB'),
                    generate_text_board('text_board_na', 'NA', '0%'),
                    generate_text_board('text_board_num','Numeric', '0%'),
                ],
                align = "start",
                justify = 'center',
                className='row_overview_textboard'
            ),
            
            dbc.Row(
                [
                    generate_datatable_with_dataframe('datatable_overview', title='Feature Overview'),

                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'

            ),


            dbc.Row(
                [
                    # histogram
                    generate_histogram_with_rangeslide('dropdown_overview_features_selection_histogramgraph', 'dropdown_overview_feature_selection_rangeslider_histogram', 'figure_overview_histogram', 'rangeslider_overview_value_constraint_histogram', 'Histogram'),
                    # line graph
                    generate_line_with_rangeslide('dropdown_overview_features_selection_linegraph', 'dropdown_overview_feature_selection_rangeslider_linegraph', 'figure_overview_linegraph', 'rangeslider_overview_value_constraint_linegraph', 'Line Plot'),
                    
                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'
            ),


            dbc.Row(
                [
                    # generate correlation heatmap
                    generate_correlation_heatmap('dropdown_overview_feature_selection_heatmap', 'figure_overview_heatmap', 'Correlations'),
                    # scatter graph
                    generate_scatter_with_rangeslide('dropdown1_overview_feature_selection_scattergraph', 'dropdown2_overview_feature_selection_scattergraph', 'figure_overview_scattergraph', 'rangeslider_overview_value_constraint_scattergraph', 'Scatter Plot'),

                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'

            ),
        ],
    )


def generate_text_board(id, text, value):
    return dbc.Col(
        dbc.Card(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.CardImg(src="/assets/img/shape.svg", className='img-fluid rounded-start'),
                            width = "auto",
                        ),
                        dbc.Col(
                            dbc.CardBody(
                                [
                                    html.H6(id=id, children=str(value), className='text_board_font1'), 
                                    html.P(str(text), className='text_board_font2')
                                ]
                            ),
                            width = "auto"
                        )
                    ],
                ),
            ],
            className='text_board_card',
        ),
        width=3,
    )




##### histogram
def generate_histogram_with_rangeslide(id_fs1, id_fs2, id_figure, id_slider, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_features = options_all[:2]
        
        
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = list(df_num.columns)
        value_index= options_num[0]
        
        min_first_num = df_num.iloc[:, 0].min()
        max_first_num = df_num.iloc[:, 0].max()
    else:
        options_all = [] 
        value_features = None
        options_num = []
        value_index = None
        min_first_num = 0
        max_first_num = 1
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close('button_overview_histogram_close', 'button_overview_histogram_fullscreen', title, 'card_overview_histogram'),
                dbc.CardBody(
                    [
                        html.Div([
                            dcc.Dropdown(
                                id = id_fs1,
                                options=options_all,
                                value=value_features,
                                multi=True,
                                className='dropdown_overview_multi_feature',
                                clearable=False
                            ),
                        ]),
                        dcc.Graph(
                            id = id_figure,
                            className='figure_overview'
                        ),
                        html.Div(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id = id_fs2,
                                        options=options_num,
                                        value=value_index,
                                        clearable=False
                                    ),
                                    width = 4
                                ),
                                dbc.Col(
                                    html.Div(
                                        dcc.RangeSlider(
                                            id = id_slider,
                                            min= min_first_num,
                                            max= max_first_num,
                                            marks = {i: {'label': str(round(i))} for i in np.arange(min_first_num, max_first_num, (max_first_num-min_first_num)/5)},
                                            tooltip={"placement": "top", "always_visible": False},
                                            value=[min_first_num, max_first_num],
                                        ),
                                    ),
                                    width = 8
                                )

                            ],
                            className = 'row',
                        )

                    ],
                )
            ],
        ),
        width = 6,
        className='card_container',
        id = 'card_overview_histogram'
    )
    return layout


#### line graph
def generate_line_with_rangeslide(id_fs1, id_fs2, id_figure, id_slider, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_features = options_all[:2]
        
        
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = list(df_num.columns)
        value_index= options_num[0]
        
        min_first_num = df_num.iloc[:, 0].min()
        max_first_num = df_num.iloc[:, 0].max()
    else:
        options_all = [] 
        value_features = None
        options_num = []
        value_index = None
        min_first_num = 0
        max_first_num = 1
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close('button_overview_linegraph_close', 'button_overview_linegraph_fullscreen', title, 'card_overview_linegraph'),
                dbc.CardBody(
                    [
                        html.Div([
                            dcc.Dropdown(
                                id = id_fs1,
                                options=options_all,
                                value= value_features,
                                multi=True,
                                className='dropdown_overview_multi_feature',
                                clearable=False
                            ),
                        ]),
                        dcc.Graph(
                            id = id_figure,
                            className='figure_overview'
                        ),
                        html.Div(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id = id_fs2,
                                        options=options_num,
                                        value=value_index,
                                        clearable=False
                                    ),
                                    width = 4,
                                ),
                                dbc.Col(
                                    html.Div(
                                        dcc.RangeSlider(
                                            id = id_slider,
                                            min= min_first_num,
                                            max= max_first_num,
                                            marks = {i: {'label': str(round(i))} for i in np.arange(min_first_num, max_first_num, (max_first_num-min_first_num)/5)},
                                            tooltip={"placement": "top", "always_visible": False},
                                            value=[min_first_num, max_first_num],
                                        ),
                                    ),
                                    width = 8
                                ),
                            ],
                            className = 'row',
                        )

                    ],
                    #className="four columns",
                )
            ],
        ),
        width = 6,
        className='card_container',
        id = 'card_overview_linegraph'  
    )
    return layout



#### scatter plot
def generate_scatter_with_rangeslide(id_fs1, id_fs2, id_figure, id_slider, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = list(df_num.columns)
        value_num_1 = options_num[0]
        value_num_2 = options_num[1]
        
        min_first_num = df_num.iloc[:, 0].min()
        max_first_num = df_num.iloc[:, 0].max()
    else:
        options_num = []
        value_num_1 = None
        value_num_2 = None
        min_first_num = 0
        max_first_num = 1
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close('button_overview_scatter_close', 'button_overview_scatter_fullscreen', title, 'card_overview_scatter'),
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div([
                                    dcc.Dropdown(
                                        id = id_fs1,
                                        options=options_num,
                                        value= value_num_1,
                                        className='dropdown_overview_single_feature',
                                        clearable=False
                                    ),
                                ]),
                                html.Div([
                                    dcc.Dropdown(
                                        id = id_fs2,
                                        options=options_num,
                                        value= value_num_2,
                                        className='dropdown_overview_single_feature',
                                        clearable=False
                                    ),
                                ]),
                            ],
                            className='div_overview_single_feature'
                        ),
                        dcc.Graph(
                            id = id_figure,
                            className='figure_overview'
                        ),
                        html.Div(
                            [
                                dbc.Col(
                                    html.Div(
                                        dcc.RangeSlider(
                                            id = id_slider,
                                            min= min_first_num,
                                            max= max_first_num,
                                            marks = {i: {'label': str(round(i))} for i in np.arange(min_first_num, max_first_num, (max_first_num-min_first_num)/5)},
                                            tooltip={"placement": "top", "always_visible": False},
                                            value=[min_first_num, max_first_num],
                                        ),
                                    ),
                                    width = 16
                                )
                            ],
                            className = 'row',
                        )

                    ],
                )
            ],
        ),
        width = 6,
        className='card_container',
        id = 'card_overview_scatter'  
    )
    return layout



# generate describe table_data
def generate_datatable_with_dataframe(id, title=''):
    if table_data.DF_RAW is not None:
        df = analyse_df(table_data.DF_RAW)
        cols = [{"name": i, "id": i} for i in OVERVIEW_COLUMNS]
        data = df.to_dict('records')
    else:
        cols = [{"name": i, "id": i} for i in OVERVIEW_COLUMNS]
        data = None
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close('button_overview_datatable_close', 'button_overview_datatable_fullscreen', title, 'card_overview_datatable'),
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id=id,
                            columns=cols,
                            data = data,
                            filter_action='native',
                            row_deletable=True,
                            editable=True,
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
                    ],
                    className='card_body_one_col',
                    id='cardbody_overview_datatable'
                )
            ],
        ),
        width=12,
        className='card_container',
        id='card_overview_datatable'        
    )
    return layout

def generate_correlation_heatmap(id_dropdown, id_figure, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        df_num = df.select_dtypes(include=NUMERICS)
        options_num =list(df_num.columns)
        value_num = options_num[:5]
        
        min_first_num = df_num.iloc[:, 0].min()
        max_first_num = df_num.iloc[:, 0].max()
    else:
        options_num = []
        value_num = None
        min_first_num = 0
        max_first_num = 1
    
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close('button_overview_heatmap_close', 'button_overview_heatmap_fullscreen', title, 'card_overview_heatmap'),
                dbc.CardBody(
                    [
                        html.Div([
                            dcc.Dropdown(
                                id = id_dropdown,
                                options=options_num,
                                value= value_num,
                                multi=True,
                                className='dropdown_overview_multi_feature',
                                clearable=False
                            ),
                        ]),
                        dcc.Graph(id=id_figure,className='figure_overview')
                    ],
                )
            ],
        ),
        width=6,
        className='card_container',
        id = 'card_overview_heatmap'  
    )
    return layout


def generate_line_figure_under_constraint(cols_h, col_c, value_min_c, value_max_c):
    """
    filter data and create histogram, only first 5 will be shown
    cols_h: feature that show in the histgramgraph
    col_c: feature name that used to filter the data
    value_min_c: min acceptable value
    value_max_c: max acceptable value
    """
    # filter data
    df = compute_plot(table_data.DF_RAW, col_c, cols_h, value_min_c, value_max_c, reset_index=True)
    
    # draw Figure
    figure = get_overview_line_plot(df, cols_h)
    
    return figure 


def generate_scatter_figure_under_constraint(col1, col2, value_min_c, value_max_c):
    """
    filter data and create histogram, only first 5 will be shown
    cols: feature that show in the histgramgraph
    value_min_c: min acceptable value
    value_max_c: max acceptable value
    """
    df = compute_scatter(table_data.DF_RAW, col1, value_min_c, value_max_c)
    figure = get_overview_scatter_plot(df, col1, col2)
    return figure

def generate_heatmap_figure_with_dataframe(df):
    corr = analyse_correlation(df)
    figure = get_overview_heatmap(corr)
    return figure


############# util
def add_cardheader_for_fullscreen_and_close(id_close, id_fullscreen, title, id_target=None):
    layout = dbc.CardHeader(
        [
            dbc.Row(
                [
                    dbc.Col(html.H5(title, className='h5'), width=11),
                    dbc.Col(
                        [
                            dbc.Button(
                                html.Img(src="/assets/img/fullscreen.png", className='btn_img'),
                                id=id_fullscreen,
                                color="#FDFCFC",
                                style={'display': 'block'}
                            ),
                            dbc.Button(
                                html.Img(src="/assets/img/close.png", className='btn_img'),
                                id=id_close,
                                color="#FDFCFC",
                                style={'display': 'block'}
                            )
                        ],
                        width=1,
                        style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}
                    ),
                ],
                className='card_header_fullscreen_and_close'
            )
        ],
    )
    
    if id_target:
        # regist callback
        @app.callback(
            Output(id_target, "style"),
            Output(id_close, "style"),
            Input(id_close, 'n_clicks'),
            Input(id_fullscreen, 'n_clicks'),
            State(id_target, "style"),
            State(id_close, "style")

        )
        def fullscreen_target(n_clicks1, n_clicks2, style, style_close):
            triggered_id = ctx.triggered_id
            if triggered_id == id_close:
                if n_clicks1 and n_clicks1%2 == 1:
                    style['display'] = 'none'
                else:
                    style['display'] = 'block'
                return style
            else:
                style_card_full = {
                                    'height': '90vh',
                                    'width': 'auto',
                                    'zIndex': 998,
                                    'position': 'fixed', 'top': '55px',
                                    'bottom': 0, 'left': 0, 'right': 0,
                                    'background-color': 'white'
                                }
                style_card_small = {

                }
                style_component = {
                    'height': '89vh', 'maxHeight': '89vh',
                }
                if n_clicks2 and n_clicks2%2 == 1:
                    style_close['display'] = 'none'
                    return style_card_full, style_close
                else:
                    style_close['display'] = 'block'
                    return style_card_small, style_close

    return layout

# Define the page layout
layout = dbc.Container(
        [
            create_data_overview_panel()
        ],
        id = "data_overview_container",
        fluid = True,
        style = {"display": "none"}
    )
