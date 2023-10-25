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

# import slider marks
from view.page_helper_components.sliders import get_slider_marks

# Define the page layout
def create_data_overview_panel():
    return html.Div(
        [
            dbc.Row(
                [
                    generate_text_board('text_board_shape', '(#Rows, #Features)', '(0,0)'),
                    generate_text_board('text_board_memory', 'Memory used', '0MB'),
                    generate_text_board('text_board_na', 'Missing', '0%'),
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
                    generate_lineplot('dropdown_overview_features_selection_linegraph', 'dropdown_overview_feature_selection_rangeslider_linegraph', 'dropdown_overview_target_selection_linegraph', 'dropdown_overview_class_selection_linegraph', 'figure_overview_linegraph', 'Line Plot'),
                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'

            ),


            dbc.Row(
                [
                    # histogram
                    generate_histogram('dropdown_overview_features_selection_histogram', 'dropdown_overview_target_selection_histogram', 'dropdown_overview_class_selection_histogram', 'figure_overview_histogram', 'Histogram'),
                    # line graph
                    generate_violinplot('dropdown_overview_features_selection_violinplot', 'dropdown_overview_feature_selection_rangeslider_violinplot', 'dropdown_overview_target_selection_violinplot', 'dropdown_overview_class_selection_violinplot', 'figure_overview_violinplot', 'Violin Plot'),
                    
                ],
                align = "start",
                justify = 'center',
                className='row_overview_plots'
            ),


            dbc.Row(
                [
                    # generate correlation heatmap
                    generate_correlation_heatmap('dropdown_overview_feature_selection_heatmap', 'dropdown_overview_target_selection_heatmap', 'dropdown_overview_class_selection_heatmap', 'figure_overview_heatmap', 'Correlations'),
                    # scatter graph
                    generate_scatter('dropdown1_overview_feature_selection_scattergraph', 'dropdown2_overview_feature_selection_scattergraph', 'dropdown_overview_target_selection_scattergraph', 'dropdown_overview_class_selection_scattergraph', 'figure_overview_scattergraph','Scatter Plot'),

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
                                    dcc.Loading(
                                        id=id,
                                        children=html.H6(children=str(value), className='text_board_font1'), 
                                    ),
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

#### violinplot
def generate_lineplot(id_fs1, id_fs2, id_fs3, id_fs4, id_figure, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_features = options_all[:2]
        value_target = options_all[0]
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = ['index_auto'] + list(df_num.columns)
        value_index= options_num[0]
    else:
        options_all = [] 
        value_features = None
        value_target = None
        options_num = []
        value_index = None
        
    tooltip = "Plot the features in a line plot. If your data contains a class feature you can filter the data for a specific class to only show data points that belong to this class."
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close(title, 'img_overview_lineplot', tooltip),
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
                        html.Div(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Target:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs3,
                                            options=options_all,
                                            value=value_target,
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target_large',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Class:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs4,
                                            options=['ALL'] + options_all,
                                            value='ALL',
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target_large',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                            ],
                            className='div_overview_single_feature'
                        ),
                        dcc.Loading(
                            id=id_figure,
                            type="default",
                            children=dcc.Graph(className='figure_overview')
                        ),
                        html.Div(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Index:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id=id_fs2,
                                            options=options_num,
                                            value=value_index,
                                            clearable=False,
                                            className='dropdown_overview_index',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width=6,
                                )
                            ],
                            className='row',
                        )

                    ],
                    #className="four columns",
                )
            ],
        ),
        width = 12,
        className='card_container',
        id = 'card_overview_linegraph'  
    )
    return layout


##### histogram
def generate_histogram(id_fs1, id_fs3, id_fs4, id_figure, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_features = options_all[0]
        value_target = options_all[0]
        
        
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = ['index_auto'] + list(df_num.columns)
        value_index= options_num[0]
        
        min_first_num = df_num.iloc[:, 0].min()
        max_first_num = df_num.iloc[:, 0].max()
    else:
        options_all = [] 
        value_features = None
        value_target = None
        options_num = []
        value_index = None
        min_first_num = 0
        max_first_num = 1
    
    tooltip = "Plot a the frequency of feature values of a feature in a histogram. If your data contains a class feature you can filter the data for a specific class to only show data points that belong to this class."
    
    marks = get_slider_marks((min_first_num, max_first_num))
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close(title, 'img_overview_histogramm', tooltip),
                dbc.CardBody(
                    [
                        html.Div([
                            dcc.Dropdown(
                                id = id_fs1,
                                options=options_all,
                                value=value_features,
                                multi=False,
                                className='dropdown_overview_multi_feature',
                                clearable=False
                            ),
                        ]),
                        html.Div(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Target:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs3,
                                            options=options_all,
                                            value=value_target,
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Class:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs4,
                                            options=['ALL'] + options_all,
                                            value='ALL',
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                            ],
                            className='div_overview_single_feature'
                        ),
                        dcc.Loading(
                            id=id_figure,
                            type="default",
                            children=dcc.Graph(className='figure_overview')
                        ),

                    ],
                )
            ],
        ),
        width = 6,
        className='card_container',
        id = 'card_overview_histogram'
    )
    return layout


#### violinplot
def generate_violinplot(id_fs1, id_fs2, id_fs3, id_fs4, id_figure, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_features = options_all[:2]
        value_target = options_all[0]
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = ['index_auto'] + list(df_num.columns)
        value_index= options_num[0]
    else:
        options_all = [] 
        value_features = None
        value_target = None
        options_num = []
        value_index = None
        
    tooltip = "Plot the distribution of the feature values in a violin plot. If your data contains a class feature you can filter the data for a specific class to only show data points that belong to this class."
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close(title, 'img_overview_violinplot', tooltip),
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
                        html.Div(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Target:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs3,
                                            options=options_all,
                                            value=value_target,
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Class:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs4,
                                            options=['ALL'] + options_all,
                                            value='ALL',
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                            ],
                            className='div_overview_single_feature'
                        ),
                        dcc.Loading(
                            id=id_figure,
                            type="default",
                            children=dcc.Graph(className='figure_overview')
                        ),

                    ],
                    #className="four columns",
                )
            ],
        ),
        width = 6,
        className='card_container',
        id = 'card_overview_violinplot'  
    )
    return layout



#### scatter plot
def generate_scatter(id_fs1, id_fs2, id_fs3, id_fs4, id_figure, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_target = options_all[0]
        df_num = df.select_dtypes(include=NUMERICS)
        options_num = list(df_num.columns)
        value_num_1 = options_num[0]
        value_num_2 = options_num[1]
    else:
        options_all = []
        value_target = None
        options_num = []
        value_num_1 = None
        value_num_2 = None
        
    tooltip = "Plot two features in a scatter plot. If your data contains a class feature you can filter the data for a specific class to only show data points that belong to this class. Or you can plot all classes and the datapoints are colored in different according to the corresponding class."    
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close(title, 'img_overview_scatter', tooltip),
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
                        html.Div(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Target:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs3,
                                            options=options_all,
                                            value=value_target,
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Class:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs4,
                                            options=['ALL'] + options_all,
                                            value='ALL',
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                            ],
                            className='div_overview_single_feature'
                        ),
                        dcc.Loading(
                            id=id_figure,
                            type="default",
                            children=dcc.Graph(className='figure_overview')
                        ),

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
     
    tooltip = "In this table you can delete features by clicking the cross beside the feature. You can also use the search function (first row under the header) to filter for values. In case of numeric values you can also use '<', '>','<=', '>=' and '!='. E.g. use '< 60.8' to filter for values less than 60.8."
        
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close(title, 'img_overview_datatable', tooltip),
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id=id,
                            columns=cols,
                            data = data,
                            filter_action='native',
                            row_deletable=True,
                            editable=False,
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

def generate_correlation_heatmap(id_dropdown, id_fs2, id_fs3, id_figure, title=''):
    if table_data.DF_RAW is not None:
        df = table_data.DF_RAW
        options_all = list(df.columns)
        value_target = options_all[0]
        df_num = df.select_dtypes(include=NUMERICS)
        options_num =list(df_num.columns)
        value_num = options_num[:5]
        
        min_first_num = df_num.iloc[:, 0].min()
        max_first_num = df_num.iloc[:, 0].max()
    else:
        options_all = []
        options_num = []
        value_num = None
        value_target = None
        min_first_num = 0
        max_first_num = 1
        
    tooltip = "Plot the correlations between the different features. If your data contains a class feature you can filter the data for a specific class to only show data points that belong to this class."  
    
    layout = dbc.Col(
        dbc.Card(
            [
                add_cardheader_for_fullscreen_and_close(title, 'img_overview_heatmap', tooltip),
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
                        html.Div(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Target:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs2,
                                            options=options_all,
                                            value=value_target,
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Class:", className='text_overview_index', style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                        dcc.Dropdown(
                                            id = id_fs3,
                                            options=['ALL'] + options_all,
                                            value='ALL',
                                            clearable=False,
                                            multi=False,
                                            className='dropdown_overview_target',
                                            style={'display': 'inline-block', 'vertical-align': 'middle'}
                                        ),
                                    ],
                                    width = 6
                                ),
                            ],
                            className='div_overview_single_feature'
                        ),
                        dcc.Graph(id=id_figure,className='figure_overview'),
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
def add_cardheader_for_fullscreen_and_close(title, id_target="", tooltip=""):
    if tooltip == "":
        layout = dbc.CardHeader(
            [
                dbc.Row(
                    [
                        dbc.Col(html.H5(title, className='h5'), width=11),
                    ],
                    className='card_header_fullscreen_and_close'
                )
            ],
        )
    else:
        layout = dbc.CardHeader(
            [
                html.Img(id=id_target, src="/assets/img/tooltip.png", className='tooltip_img'),
                dbc.Tooltip(tooltip, target=id_target),
                html.H5(title, className='h5'),
            ],
        )
    
    if id_target:
        pass
        # regist callback
        #@app.callback(
        #    Output(id_target, "style"),
        #    Output(id_close, "style"),
        #    Input(id_close, 'n_clicks'),
        #    Input(id_fullscreen, 'n_clicks'),
        #    State(id_target, "style"),
        #    State(id_close, "style")

        #)
        #def fullscreen_target(n_clicks1, n_clicks2, style, style_close):
        #    triggered_id = ctx.triggered_id
        #    if triggered_id == id_close:
        #        if n_clicks1 and n_clicks1%2 == 1:
        #            style['display'] = 'none'
        #        else:
        #            style['display'] = 'block'
        #        return style
        #    else:
        #        style_card_full = {
        #                            'height': '90vh',
        #                            'width': 'auto',
        #                            'zIndex': 998,
        #                            'position': 'fixed', 'top': '55px',
        #                            'bottom': 0, 'left': 0, 'right': 0,
        #                            'background-color': 'white'
        #                        }
        #        style_card_small = {

        #        }
        #        style_component = {
        #            'height': '89vh', 'maxHeight': '89vh',
        #        }
        #        if n_clicks2 and n_clicks2%2 == 1:
        #            style_close['display'] = 'none'
        #            return style_card_full, style_close
        #        else:
        #            style_close['display'] = 'block'
        #            return style_card_small, style_close

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
