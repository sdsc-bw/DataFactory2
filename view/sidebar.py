import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from view.app import app

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "33rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll"
}



# ---------------------------------- SUBMENU 1 ---------------------------------------

button_dl = dbc.Button(
    "Data Loading",
    outline=False,
    #active=True,
    color="primary",
    href="/page-1/0",
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_do = dbc.Button(
    "Data Overview",
    id="button_overview",
    outline=False,
    #active=True,
    color="primary",
    href="/page-1/1",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)


button_cf = dbc.Button(
    "Categorial Features",
    id="button_categorical",
    outline=False,
    #active=True,
    color="primary",
    href="/page-1/2",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_na = dbc.Button(
    "NA Values",
    id="button_na_values",
    outline=False,
    #active=True,
    color="primary",
    href="/page-1/3",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_od = dbc.Button(
    "Outlier Detection",
    id="button_outlier",
    outline=False,
    #active=True,
    color="primary",
    href="/page-1/4",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

submenu_1 = [
    html.Li(
        # use Row and Col components to position the chevrons
        dbc.Row(
            [
                dbc.Col(html.I(className="fa fa-database fa-lg",style={"margin-right":"10px"}), width="auto",),

                    
                dbc.Col(
                    html.Span(
                        "Data Selection",
                        style={"font-family": "Times New Roman","font-size": "22pt"}
                    ),
                    width="auto"
                ),
                #dbc.Col("Data Selection"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right fa-lg", style={"padding-top": "18px"}),
                    width="auto",
                ),
            ],
            #className="my-1",
            align="center",
        ),
        style={"cursor": "pointer","padding": "1rem 0rem 0rem 0rem"},
        id="submenu-1",
    ),
    # we use the Collapse component to hide and reveal the navigation links
    dbc.Collapse(
        dbc.Nav(
            #[
            #    dbc.NavItem(dbc.NavLink("Page 3.1", href="/page-3/1")),
            #    dbc.NavItem(dbc.NavLink("Page 3.2", href="/page-3/2")),
            #],
            [dbc.NavItem(button_dl), dbc.NavItem(button_do), dbc.NavItem(button_cf), dbc.NavItem(button_na), dbc.NavItem(button_od)], 
            #navbar=True,
            vertical=True,
            style={"float":"right", "margin-right":"5rem"}
            #pills=True
        ),
        id="submenu-1-collapse",
    ),
]


# ---------------------------------- SUBMENU 2 ---------------------------------------

button_td = dbc.Button(
    "Transformation",
    outline=False,
    #active=True,
    color="primary",
    href="/page-2/1",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)


# button_id = dbc.Button(
#     "Image Data",
#     outline=False,
#     #active=True,
#     color="primary",
#     href="/page-2/2",
#     #id="gh-link",
#     style={
#         "text-transform": "none", 
#         'margin-top': "0.5rem",
#         "font-family": 
#         "Times New Roman","font-size": "16pt"},
# )


button_pm = dbc.Button(
    "Process Data",
    outline=False,
    #active=True,
    color="primary",
    href="/page-2/5",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)


button_pd = dbc.Button(
    "Position Data",
    outline=False,
    #active=True,
    color="primary",
    href="/page-2/3",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_id = dbc.Button(
    "Image Data",
    outline=False,
    #active=True,
    color="primary",
    href="/page-2/4",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_ts = dbc.Button(
    "Transformation",
    id='button_ts',
    outline=False,
    #active=True,
    color="primary",
    href="/page-2/2",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

submenu_2 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col(html.I(className="fa fa-line-chart fa-lg",style={"margin-right":"5px"}), width="auto",),

                    
                dbc.Col(
                    html.Span(
                        "Data Exploration",
                        style={"font-family": "Times New Roman","font-size": "22pt"}
                    ),
                    width="auto"
                ),
                #dbc.Col("Data Selection"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right fa-lg", style={"padding-top": "18px"}),
                    width="auto",
                ),
            ],
            #className="my-1",
            align="center",
        ),        
        style={"cursor": "pointer","padding": "1rem 0rem 0rem 0rem"},
        id="submenu-2",
    ),
    dbc.Collapse(
        #[
        #    dbc.NavLink("Page 2.1", href="/page-2/1"),
        #    dbc.NavLink("Page 2.2", href="/page-2/2"),
        #],
        dbc.Nav(
            #[
            #    dbc.NavItem(dbc.NavLink("Page 3.1", href="/page-3/1")),
            #    dbc.NavItem(dbc.NavLink("Page 3.2", href="/page-3/2")),
            #],
            #[dbc.NavItem(button_td), dbc.NavItem(button_pm),dbc.NavItem(button_pd), dbc.NavItem(button_id), dbc.NavItem(button_ts)],
            [dbc.NavItem(button_ts)], 
            #navbar=True,
            vertical=True,
            style={"float":"right", "margin-right":"8rem"}
            #style={'margin-top': 1}
            #pills=True
        ),
        id="submenu-2-collapse",
    ),
]


# ---------------------------------- SUBMENU 3 ---------------------------------------
button_sc = dbc.Button(
    "Supervised Classification",
    id='button_sc',
    outline=False,
    #active=True,
    color="primary",
    href="/page-3/1",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_sr = dbc.Button(
    "Supervised Regression",
    id='button_sr',
    outline=False,
    #active=True,
    color="primary",
    href="/page-3/2",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

button_usl = dbc.Button(
    "Unsupervised Learning",
    id='button_usl',
    outline=False,
    #active=True,
    color="primary",
    href="/page-3/3",
    disabled=True,
    #id="gh-link",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "16pt"},
)

submenu_3 = [
    html.Li(

        dbc.Row(
            [
                dbc.Col(html.I(className="fa fa-cogs fa-lg",style={"margin-right":"5px"}), width="auto",),

                    
                dbc.Col(
                    html.Span(
                        "Data Analysis",
                        style={"font-family": "Times New Roman","font-size": "22pt"}
                    ),
                    width="auto"
                ),
                #dbc.Col("Data Selection"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right fa-lg", style={"padding-top": "18px"}),
                    width="auto",
                ),
            ],
            #className="my-1",
            align="center",
        ),   
        style={"cursor": "pointer","padding": "1rem 0rem 0rem 0rem"},
        id="submenu-3",
    ),
    dbc.Collapse(
        dbc.Nav(
            #[
            #    dbc.NavItem(dbc.NavLink("Page 3.1", href="/page-3/1")),
            #    dbc.NavItem(dbc.NavLink("Page 3.2", href="/page-3/2")),
            #],
            [dbc.NavItem(button_sc), dbc.NavItem(button_sr)], 
            #navbar=True,
            vertical=True,
            style={"float":"right", "margin-right":"1rem"}
            #pills=True
        ),
        id="submenu-3-collapse",
    ),
]




button_report_bugs = dbc.Button(
    "Contact",
    outline=True,
    #active=True,
    color="primary",
    #href="/page-3/2",
    id="contact",
    style={
        "text-transform": "none", 
        'margin-top': "0.5rem",
        "font-family": 
        "Times New Roman","font-size": "10pt"},
)



def sidebar():
    layout = html.Div(
        [
            #html.H2("WEFA Inotec", className="display-7"),
            html.A(
                href = "https://wefa.com/",
                children=[
                    html.Img(src=app.get_asset_url('./img/logo.png'), style={"width": "30rem"}),
                ]
            ),
            
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "borderColor": "black", "borderStyle":"solid"}),
            html.P(
                "Ihr innovativer Partner für Extrusionswerkzeuge", 
                style={"font-family": "Times New Roman","font-size": "16pt","font-style":"italic"}
            ),
            html.Hr(style={'borderWidth': "0.3vh", "width": "100%", "borderColor": "black", "borderStyle":"solid"}),
            dbc.Nav(
                submenu_1 + submenu_2 + submenu_3, 
                vertical=True,
                pills=False,
		    	#style = {"padding": "50px 50px 50px 50px"}
	    	),
            #html.Div(submenu_1 + submenu_2 + submenu_3),
            #html.Div(submenu_2),
            #html.Div(submenu_3),
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=app.get_asset_url('./img/kit.png'),style={"width": "10rem", "display": "flex",
        "justify-content": "center"})),
                        dbc.Col(button_report_bugs,style={"display": "flex",
        "justify-content": "center"}),
                    ]
                ),
                style={"margin":"2rem"},
            ),
        ],
        style=SIDEBAR_STYLE,
        id="sidebar",
    )
    return layout
