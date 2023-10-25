import dash
import dash_bootstrap_components as dbc
import os

assets_path = os.getcwd() +'/assets'

app = dash.Dash(
    __name__,
    #suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    #prevent_initial_callbacks='initial_duplicate',
    assets_folder=assets_path,
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)