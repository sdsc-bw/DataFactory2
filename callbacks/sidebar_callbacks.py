import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

# Import app
from view.app import app


# this function is used to toggle the is_open property of each Collapse
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# this function applies the "open" class to rotate the chevron
def set_navitem_class(is_open):
    if is_open:
        return "open"
    return ""


for i in [1, 2, 3]:
    app.callback(
        Output(f"submenu-{i}-collapse", "is_open"),
        [Input(f"submenu-{i}", "n_clicks")],
        [State(f"submenu-{i}-collapse", "is_open")],
    )(toggle_collapse)

    app.callback(
        Output(f"submenu-{i}", "className"),
        [Input(f"submenu-{i}-collapse", "is_open")],
    )(set_navitem_class)



@app.callback(
    #Output("page-content", "children"),
    [
        Output("data_loading_container", "style"),
        Output("data_overview_container", "style"),        
        Output("data_categorical_container", "style"),
        Output("data_na_value_container", "style"),
        Output("data_outlier_detection_container", "style"),
#        Output("data_transformation_table_data_container", "style"),
        Output("data_transformation_time_series_container", "style"),
#        Output("data-visualization-container", "style"),
        Output("data_supervised_classification_container", "style"),
        Output("data_supervised_regression_container", "style"),
#        Output("data-unsupervised-learning-container", "style"),
    ],
    [Input("url", "pathname")],
)
def render_page_content(pathname):
    print(pathname)
    on = {"display": "block"}
    off = {"display": "none"}
    if pathname in ["/", "/page-1/0"]:
        return on, off, off, off, off, off, off, off
    elif pathname in ["/", "/page-1/1"]:
        return off, on, off, off, off, off, off, off
    elif pathname in ["/", "/page-1/2"]: # categorical feature
        return off, off, on, off, off, off, off, off
    elif pathname == "/page-1/3":  # na value
        return off, off, off, on, off, off, off, off
    elif pathname == "/page-1/4": # outlier
        return off, off, off, off, on, off, off, off
#    elif pathname == "/page-2/1": # table data
#        return off, off, off, off, off, on, off, off
    elif pathname == "/page-2/2": # ts
        return off, off, off, off, off, on, off, off

#    elif pathname == "/page-2/3":
#        return html.P("Oh cool, this is page 2.3!")
#    elif pathname == "/page-2/4":
#        return html.P("No way! This is page 2.4!")
#   elif pathname == "/page-2/5":
#        return html.P("No way! This is page 2.5!")        
    elif pathname == "/page-3/1": # data supervised classification training
        return off, off, off, off, off, off, on, off
    elif pathname == "/page-3/2": # data supervised classification training
        return off, off, off, off, off, off, off, on
#    elif pathname == "/page-3/2": # data unsupervised training
#        return off, off, off, off, off, off
    elif pathname in  ["/regression","/classification"]:
        return dash.no_update
    else:
        return on, off, off, off, off, off, off



        
        