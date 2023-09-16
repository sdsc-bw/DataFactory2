import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

# Connect to main app.py file
from view.app import app

# Connect the navbar to the index
from view import sidebar

# import callbacks
from callbacks import sidebar_callbacks

# import pages
from view import page_data_loading, page_overview, page_categorical_feature, page_na_value, page_outlier_detection, page_transformation_table_data, page_transformation_time_series, page_supervised_classification, page_supervised_regression

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "32rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


sidebar_ = sidebar.sidebar()


content = html.Div(
    [
        page_data_loading.layout,
        page_overview.layout,
        page_categorical_feature.layout,
        page_na_value.layout,
        page_outlier_detection.layout,
        page_transformation_time_series.layout,
        page_supervised_classification.layout,
        page_supervised_regression.layout,

    ],
    id="page_content", style=CONTENT_STYLE
)

app.layout = html.Div([dcc.Location(id="url"), sidebar_, content])




if __name__ == "__main__":
    app.run_server()
