import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import random

# import util
from methods.util import is_close

from methods.data_exploration.analyse import NUMERICS

PLOTS = ['Line Plot', 'Histogram', 'Correlations', 'Scatter Plot', 'Violin Plot']

def get_overview_histogram_plot(df, col):
    if type(col) == list:
        col = col[0]
        
    figure = px.histogram(df, x=col)
    
    figure.update_layout(
    xaxis_title=col,
    yaxis_title='frequency',
    bargap=0.1)
    
    return figure

def get_overview_line_plot(df, cols, index='index'):
    if type(cols) == str:
        cols = [cols]
        
    df = df.select_dtypes(include=NUMERICS)
    
    figure = go.Figure()
    
    if index == 'index':
        x = df['index']
    else:
        x = df.index
    
    for i, col in enumerate(cols):
        figure.add_trace(
            go.Scatter(
                x=x, 
                y=df[col], 
                name=col)
        )
        
    figure.update_yaxes(title_text="features")
    figure.update_xaxes(title_text=index)
        
    return figure

def get_overview_violin_plot(df, cols, index='index'):
    if type(cols) == str:
        cols = [cols]
        
    figure = go.Figure()
    
    for index, col in enumerate(cols):
        figure.add_trace(go.Violin(y=df[col], box_visible=True, meanline_visible=True, opacity=0.6, name=col))
        
    figure.update_xaxes(title_text='features')
    figure.update_yaxes(title_text='distribution')
        
    return figure

def get_overview_scatter_plot(df, col1, col2, target=None):
    if target is None:
        df = df.select_dtypes(include=NUMERICS)
        figure = px.scatter(df, x=col1, y=col2) 
    else:
        random.seed(42)
        
        unique_classes = df[target].unique().tolist()
        
        # Generate a random color for each class
        class_colors = {cls: f'{str(cls)}' for cls in unique_classes}
        
        # Map class labels to colors for each data point
        df['color'] = df[target].map(class_colors)

        # Create the scatterplot using Plotly Express
        figure = px.scatter(df, x=col1, y=col2, color='color')

        # Customize the legend
        figure.update_layout(legend_title_text=target)  # Set the legend title to the target column name
        
        # Update the legend labels to show class mappings
        legend_labels = {cls: f'{cls}: {class_colors[cls]}' for cls in unique_classes}
        #figure.update_layout(legend=dict(title_text=target, title=list(legend_labels.items())))
    
    
    return figure

def get_overview_heatmap(df):
    
    figure = px.imshow(df)
    
    return figure
    

def get_numeric_categorical_ratio_plot(num_num, num_cat):
    figure = go.Figure()
    
    figure.add_trace(
        go.Bar(
            y=[''],
            x=[num_cat],
            name='categorical features',
            orientation='h',
        )
    )
    figure.add_trace(
        go.Bar(
            y=[''],
            x=[num_num],
            name='numeric features',
            orientation='h',
        )
    )
        
    return figure

def get_categorical_feature_pie_plot(counts):
    unique_values = counts.rename_axis('unique_values').reset_index(name='counts')
    
    figure = px.pie(
        unique_values, 
        values='counts', 
        names='unique_values', 
        title='Top 15 Values'
    )
    
    return figure

def get_marks_for_rangeslider(df, col_index):
    # set new range
    value_min = df[col_index].min()
    value_max = df[col_index].max()        
    curr_min = value_min
    curr_max = value_max
    ranges = value_min, value_max
        
    # test if upper and lower border are too close
    if is_close(value_min, value_max):
        value_min = value_min - 1e-09
        value_max = value_max + 1e-09
    
    # set marker and current values of slider
    marks = {i: {'label': str(round(i)), 'style': {'color': 'black', 'font-size': '10pt'}} for i in np.arange(value_min, value_max, (value_max-value_min)/5)}
    values = [curr_min, curr_max]
    
    return value_min, value_max, curr_min, curr_max, marks, values

def get_na_bar_plot(df):    
    figure = px.bar(
        df, 
        x = 'index',
        y = '#NA'
    )
    
    figure.update_yaxes(title_text="# missing")
    
    return figure
    
def get_na_heatmap(df):
    # Create heatmap plot
    figure = px.imshow(df, color_continuous_scale='Blues')

    # Add color scale legend
    figure.update_layout(coloraxis_colorbar=dict(
        title='Missing values',
        titleside='right',
        ticks='outside',
        tickvals=[0, 255],
        ticktext=['No', 'Yes'],
        tickmode='array', # set tick mode to 'array' to use custom tick positions
        tick0=-0.5, # set the position of the first tick
        dtick=255, # set the distance between ticks
        orientation='h', # set to 'v' for a taller legend
        xpad=30, # set padding between the legend and the plot
        tickfont=dict(size=10) # set font size of tick labels
    ))
    
    return figure

def get_imputer_line_plot(df, col, colored_points, color1='blue', color2='red'):
    df = df.reset_index()
    # Create trace
    trace = go.Scatter(
        x=df.index,
        y=df[col],
        mode='markers',
        marker=dict(
            color=[color2 if i in colored_points else color1 for i in range(len(df.index))]
        )
    )

    # Create layout
    layout = go.Layout(
        title='Imputer Preview',
        xaxis=dict(title='index'),
        yaxis=dict(title=f'feature preview of {col}')
    )

    # Create figure
    figure = go.Figure(data=[trace], layout=layout)
    
    return figure

def get_violin_plot(df, cols, max_index=5):
    figure = go.Figure()
    if max_index is None:
        max_index = len(cols)
    
    for index, col in enumerate(cols):
        if index <= max_index:
            figure.add_trace(go.Violin(y=df[col],
                                    box_visible=True,
                                    meanline_visible=True,
                                    opacity=0.6, name=col,))
        else:
            figure.add_trace(go.Violin(y=df[col],
                                    box_visible=True,
                                    meanline_visible=True,
                                    opacity=0.6, name=col,
                                    visible='legendonly'))
    return figure

def get_outlier_plot(df):
    figure = px.scatter(df, x = "x", y = "y", color="Is Outlier")
    return figure

def get_prediction_plot(y_train, y_train_pred, y_test, y_test_pred, title="Original Data vs Predictions"):
    # Create a figure
    figure = go.Figure()

    # Plot y_train and y_train_pred with similar color but different brightness
    figure.add_trace(go.Scatter(x=list(range(len(y_train))),
                             y=y_train,
                             mode='lines',
                             name='Original Trainingsdata',
                             line=dict(color='rgb(31, 119, 180)')))

    figure.add_trace(go.Scatter(x=list(range(len(y_train_pred))),
                             y=y_train_pred,
                             mode='lines',
                             name='Predicted Trainingsdata',
                             line=dict(color='rgb(255, 127, 14)')))

    # Plot y_test and y_test_pred with similar color but different brightness
    figure.add_trace(go.Scatter(x=list(range(len(y_train), len(y_train) + len(y_test))),
                             y=y_test,
                             mode='lines',
                             name='Original Testdata',
                             line=dict(color='rgb(140, 186, 230)')))

    figure.add_trace(go.Scatter(x=list(range(len(y_train_pred), len(y_train_pred) + len(y_test_pred))),
                             y=y_test_pred,
                             mode='lines',
                             name='Predicted Testdata',
                             line=dict(color='rgb(255, 187, 120)')))

    # Update layout
    figure.update_layout(title=title,
                      xaxis_title='Index',
                      yaxis_title='Target Value')
    
    return figure

def get_cross_validation_plot(df, title="Results Cross Validation Scoring"):
    figure = px.bar(df, x="Fold", y="Score", title=title)
    return figure

def get_summary_plot(df):
    traces = []

    for index, row in df.iterrows():
        trace = go.Bar(x=[row['Model']], y=[row['Score']], name=row['Model'])
        traces.append(trace)

    figure = go.Figure(data=traces)
    
    return figure
    
    
    
    
    
    
    
    

