# DataFactory

The DataFactory is a data mining tool for preprocessing and analysing data. It is developed by scientsits of the [SDSC BW](https://www.sdsc-bw.de/). It offers a dashboard containing several methods to preprocesss and analyse data. It allows the user a fast and easy examination of  data according to their suitability for machine learning.


## Tutorial
In the following section we give a small introdution on the usage and capabilities of the Datafactory. DISCLAIMER: The tool is WIP and a few parts of the UI might not look the same as in the demo videos.

### Run the Tool
You can either run the `app.py` via the commandline or the `app.ipynb` as jupyter notebook. If you want to run the dashboard on a server modify the following line in the app-files:
```python
if __name__ == "__main__":
    app.run_server(host=<your server-ip>, port=<your port>)
```

### Load Data
After starting the dashboard, the first step is to load your csv-data into the dashboard. Select in the menu "Data Loading". Select the seperator of the data. You can also add an additional index column if you select 'Auto' as index, it will start at 0 and will number each datapoint.

[![Video](./demos/01_loading_data.mp4](./demos/01_loading_data.mp4)


### Get an Overview over the Data

### Preprocess Data

#### Process Categorical Data

#### Processs Missing Values

#### Detect Outlier

### Create a Dataset

### Analyse the Data
