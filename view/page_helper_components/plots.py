import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# import util
from methods.util import is_close

PLOTS = ['Line Plot', 'Histogram', 'Correlations', 'Scatter Plot', 'Violin Plot']

def get_overview_histogram_plot(df, cols):
    if type(cols) == str:
        cols = [cols]
    
    figure = go.Figure()
    
    for i, col in enumerate(cols):
        if i > 5:
            is_visible = 'legendonly'
        else:
            is_visible = True
            
        figure.add_trace(
            go.Histogram(
                x=df[col], 
                name=col, 
                visible=is_visible)
        )
        
    figure.update_layout(barmode='overlay')
    figure.update_traces(opacity=0.75)
    
    return figure

def get_overview_line_plot(df, cols, index='index'):
    if type(cols) == str:
        cols = [cols]
    
    figure = go.Figure()
    
    if index == 'index':
        x = df['index']
    else:
        x = df.index
    
    for i, col in enumerate(cols):
        if i > 5:
            is_visible = 'legendonly'
        else:
            is_visible = True
            
        figure.add_trace(
            go.Scatter(
                x=x, 
                y=df[col], 
                name=col, 
                visible=is_visible)
        )
        
    return figure

def get_overview_scatter_plot(df, col1, col2):
    figure = px.scatter(df, x=col1, y=col2)
    
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
    
    
    
    
    
    
    
    

