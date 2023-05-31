import pandas as pd

import base64
import io

from methods.util import remove_inf_columns

DATA_TYPES = ['Table Data']
SEPERATOR = {'Comma':',', 'Semicolon': ';', 'Tab': '\t', 'Space': ' '}

def parse_table_data(contents, filename, params):
    # Parse the contents of the uploaded file and return a Pandas dataframe
    sep = params['sep']
    index = params['index']
    try:
        content_type, content_string = contents.split(sep)
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            if index == 'auto':
                df = df.reset_index()
                cols = ['index_auto']
                cols.extend(df.columns[1:])
                df.columns = cols
                df['index_auto'] = df['index_auto'].astype(int)
            df = remove_inf_columns(df)     
            return df
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
            if index == 'auto':
                df = df.reset_index()
                cols = ['index_auto']
                cols.extend(df.columns[1:])
                df.columns = cols
                df['index_auto'] = df['index_auto'].astype(int)
            df = remove_inf_columns(df) 
            return df
    except Exception as e:
        print(f'Error parsing data: {e}')
        return None

    