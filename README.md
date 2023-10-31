# DataFactory

The DataFactory is a data mining tool for preprocessing and analysing data. It is developed by scientsits of the [SDSC BW](https://www.sdsc-bw.de/). It offers a dashboard containing several methods to preprocesss and analyse data. It allows the user a fast and easy examination of  data according to their suitability for machine learning.


## Tutorial
In the following section we give a small introdution on the usage and capabilities of the Datafactory. Bside that in the dashboard you can find tool tips with this symbol <img src="https://github.com/sdsc-bw/DataFactory2/blob/main/assets/img/tooltip.png" width="30"> containing additional information about the feature. We also used functions of other packages. You can click the symbol <img src="https://github.com/sdsc-bw/DataFactory2/blob/main/assets/img/link.png" width="30"> in order to be directed to the documenation of this function.

DISCLAIMER: The tool is WIP and a few parts of the UI might not look the same as in the demo videos.


### Run the Tool
You can either run the `app.py` via the commandline or the `app.ipynb` as jupyter notebook. If you want to run the dashboard on a server modify the following line in the app-files:
```python
if __name__ == "__main__":
    app.run_server(host=<your server-ip>, port=<your port>)
```

### Load Data
After starting the dashboard, the first step is to load your csv-data into the dashboard. Select in the menu "Data Loading". Browse for your data or add it with drag&drop. Then select the seperator of the data. You can also add an additional index column if you select 'Auto' as index, it will start at 0 and will number each datapoint.

TODO add demo


### Get an Overview over the Data
After you uploaded your data you can get an overview over the data in the "Data Overview" tab. 

The table shows general statistics of the given features like mean, min, max, .... In the table you can delete features by clicking the cross beside the feature. You can also use the search function (first row under the header) to filter for values. In case of numeric values you can also use '<', '>','<=', '>=' and '!='. E.g. use '< 60.8' to filter for values less than 60.8.

You can use the different types of plots (line plot, histogram, violin distribution, correlation map and scattter plot) to examine the features and gain knowledge about them. If your data contains a class feature you can filter the data for a specific class (feature value) to only show data points that belong to this class.

TODO add demo


### Clean Data
The dashboard offers several functions to clean your data from missing values and outliers. We also offer functions to convert categorical features to numeric features.

#### Process Categorical Data
You can convert categorical features to numeric features in the "Categorial Features" tab. 
There you can see how many numeric and categorical features currently exist in the dataset. Select a feature and the conversion function to convert the feature. On the left you can see a pie chart of the most frequent values of the feature.

You can only continue if you removed/converted all categorical features.

TODO add demo

#### Processs Missing Values
You can fill missing values in the "Missing Values" tab. 
There you can see a plot showing the number and position of the missing values for each feature in the dataset. Below you can fill the missing values by using one of the given methods. Select the parameters and press "Show" to display a preview of the filled features. The red dots show the filled datapoints. Then press "Apply" to adopt the changes to the dataset.

You can only continue if you removed/filled all missing values.

TODO add demo

#### Detect Outlier
You can detect an remove outliers in the "Outlier Detection" tab.
At the top you can see the violin distribution of each feature. Below you can select different outlier detection methods. Select the parameters and press "Show" to show the detected outliers (red dots). in the table below you can see a list of the outliers. You can deselect to keep the datapoints in the dataset. Then press "Remove" to delete all selected outliers from the dataset.

TODO add demo

### Create a Dataset (with Transformations)
Before you can start with the analysis you need to create a dataset in the "Transformation" tab. Therefor, press + below the feature table. Then you can name and create your new dataset. It will consist of the current features of your cleaned dataset. You can plot the features of your dataset using different plots (selected by the dropdown menu) and you can delete the features from the dataset by pressing the cross in the feature table. You can also define the index range that should be used in the analysis by using the range slider below the plot.

After creating the dataset you can either apply some transformations on the features or continue with your analysis. If you decided to add some transformation you can use the functions below. Select your parameters and press "Show" to display a preview of the edited/new features. Then press "Apply" to adopt the changes to the dataset. 

You can create multiple datasets and switch between them using the dropdown menu beside the + symbol. You can also delete datasets.
The idea of the datasets is that you can create multiple datasets with different transformations and feature sets and later compare them in the analysis.

TODO add demo

### Analyse the Data
After creating datasets you can select between "Supervised Classification" and "Supervised Regression" to perform your analysis. There you can select between different baselines and simple machine learning algorithms which can be applied on your selected dataset.

In the middle you can select beside the dataset, different algorithms, their parameters and an evaluation metric. At the end you can also name the test run (which will be displayed in the summary). You can also select the train-test split for the cross validation (the number of splits is computed accoridingly). If you are working with time series check the box "Use time series cross validation". This will ensure that the order of samples will be maintained. Press "Show" to start the cross validation training. The results are displayed on the left. Then press "Save" to compute the average score of the cross validation in the summary at the top. At the bottom, you can find a plot with the prediction of the model and the original target values. Beside that, you can find the feature importance derived from the model (not available for every model).

TODO add demo
