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
        Output(f"submenu-{i}-collapse", "is_open", allow_duplicate=True),
        [Input(f"submenu-{i}", "n_clicks")],
        [State(f"submenu-{i}-collapse", "is_open")],
        prevent_initial_call=True
    )(toggle_collapse)

    app.callback(
        Output(f"submenu-{i}", "className", allow_duplicate=True),
        [Input(f"submenu-{i}-collapse", "is_open")],
        prevent_initial_call=True
    )(set_navitem_class)



@app.callback(
    Output("data_loading_container", "style", allow_duplicate=True),
    Output("data_overview_container", "style", allow_duplicate=True),     
    Output("data_categorical_container", "style", allow_duplicate=True),
    Output("data_na_value_container", "style", allow_duplicate=True),
    Output("data_outlier_detection_container", "style", allow_duplicate=True),
    Output("data_transformation_time_series_container", "style", allow_duplicate=True),
    Output("data_supervised_classification_container", "style", allow_duplicate=True),
    Output("data_supervised_regression_container", "style", allow_duplicate=True),
    Output("button_loading", "style", allow_duplicate=True),
    Output("button_overview", "style", allow_duplicate=True),        
    Output("button_categorical", "style", allow_duplicate=True),
    Output("button_na_values", "style", allow_duplicate=True),
    Output("button_outlier", "style", allow_duplicate=True),
    Output("button_ts", "style", allow_duplicate=True),
    Output("button_sc", "style", allow_duplicate=True),
    Output("button_sr", "style", allow_duplicate=True),
    Input("url", "pathname"),
    State("button_loading", "style"),
    State("button_overview", "style"),        
    State("button_categorical", "style"),
    State("button_na_values", "style"),
    State("button_outlier", "style"),
    State("button_ts", "style"),
    State("button_sc", "style"),
    State("button_sr", "style"),
    prevent_initial_call=True
)
def render_page_content(pathname, style_loading, style_overview, style_categorical, style_na, style_outlier, style_transformation, style_sc, style_sr):
    on = {"display": "block"}
    off = {"display": "none"}
    if pathname in ["/", "/page-1/0"]:
        style_loading['background-color'] = 'navy'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return on, off, off, off, off, off, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    elif pathname in "/page-1/1":
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'navy'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return off, on, off, off, off, off, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    elif pathname == "/page-1/2": # categorical feature
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'navy'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return off, off, on, off, off, off, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    elif pathname == "/page-1/3":  # na value
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'navy'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return off, off, off, on, off, off, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    elif pathname == "/page-1/4": # outlier
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'navy'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return off, off, off, off, on, off, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    elif pathname == "/page-2/2": # ts
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'navy'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return off, off, off, off, off, on, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr    
    elif pathname == "/page-3/1": # data supervised classification training
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'navy'
        style_sr['background-color'] = 'royalblue'
        return off, off, off, off, off, off, on, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    elif pathname == "/page-3/2": # data supervised classification training
        style_loading['background-color'] = 'royalblue'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'navy'
        return off, off, off, off, off, off, off, on, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr
    else:
        style_loading['background-color'] = 'navy'
        style_overview['background-color'] = 'royalblue'
        style_categorical['background-color'] = 'royalblue'
        style_na['background-color'] = 'royalblue'
        style_outlier['background-color'] = 'royalblue'
        style_transformation['background-color'] = 'royalblue'
        style_sc['background-color'] = 'royalblue'
        style_sr['background-color'] = 'royalblue'
        return on, off, off, off, off, off, off, style_loading, style_overview, style_categorical, style_na, style_outlier,style_transformation, style_sc,style_sr

@app.callback(
    Output("button_sc", 'disabled', allow_duplicate=True), 
    Output("button_sr", 'disabled', allow_duplicate=True),
    Input('dropdown_transformation_time_series_dataset', 'options'),
    prevent_initial_call=True
)
def load_data(options):
    if len(options) < 1:
        disabled = True
    else:
        disabled = False
    return disabled, disabled   
       
        
        