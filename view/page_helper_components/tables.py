from dash import dash_table

def get_feature_importance_table(id_table, data):
    table = dash_table.DataTable(id=id_table,
                                 columns= [{"name": "Feature", "id": "Feature"}, {"name": "Importance", "id": "Importance"}],
                                 data=data,
                                 filter_action='native',
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
                                     'minWidth': "50%"
                                 },
                                ),
    
    return table
