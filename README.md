# Boston311Model

This is a class that defines a model to predict the likelihood of a 311 request in Boston being closed, given a set of input features. The model is built using `tensorflow` and `sklearn` libraries.

## Installation

To install the package, you can use pip:

```
pip install git+https://github.com/mindfulcoder49/Boston_311.git
```

## Class Attributes

- `model`: The trained machine learning model used for prediction.
- `feature_columns`: A list of feature column names used in the model.
- `feature_dict`: A dictionary where keys are the feature column names and values are lists of all the possible values for that feature.
- `train_date_range`: A dictionary with keys `"start"` and `"end"`, containing the start and end dates (as datetime objects) for the training data.
- `predict_date_range`: A dictionary with keys `"start"` and `"end"`, containing the start and end dates (as datetime objects) for the prediction data.
- `scenario`: A dictionary of scenarios that can be applied to the data during cleaning. The valid keys and values are described in the `clean_data()` method.
- `model_type`: A string indicating the type of model used (e.g., linear, logistic, etc).

## Class Methods

### `__init__(self, **kwargs)`

Constructor method for the class. Initializes all the class attributes with values passed as keyword arguments.

### `load_data(self)`

This method loads the data for training the model. It returns a Pandas DataFrame containing the data for the specified time period.

### `enhance_data(self, data)`

This method enhances the data by adding new columns. Specifically, it calculates the survival time (time between open and close date) and event (whether the request has been closed or not) columns. It also extracts the ward number from the ward column and converts the survival time to hours.

### `clean_data(self, data)`

This method drops any columns not in `feature_columns`, creates the `feature_dict`, and one-hot encodes the training data. It also applies scenarios to the data, if specified in the `scenario` attribute. The valid keys and values for the `scenario` dictionary are:

- `dropColumnValues`: A dictionary of column names and lists of values to drop.
- `keepColumnValues`: A dictionary of column names and lists of values to keep, all others being dropped.
- `dropOpen`: Drop all open cases after a certain date.
- `survivalTimeMin`: Drop all closed cases where survival time is less than a given number of seconds.
- `survivalTimeMax`: Drop all closed cases where survival time is more than a given number of seconds.

### `train_model(self, data)`

This method trains the machine learning model using the cleaned data. It splits the data into training and testing sets, scales the features, and trains the model using logistic regression.

### `predict(self, data)`

This method makes predictions on new data using the trained model. It returns a numpy array of predicted values.

## Results

The `results` folder contains Jupyter notebooks used for developing the code in this repository.

### Table of Contents

| Number | Notebook Name | Description |
| ------ | ------------- | ----------- |
| 1 | Boston311 | Initial Data exploration and model prototyping |
| 2 | Boston311_v2 | Creating initial data cleaning functions for linear and logistic models |
| 3 | Boston311_v3 | Exploring categorical outliers in the data |
| 4 | Boston311_v4 | List all to-dos and questions, and finally train our models on all the data |
| 5 | Boston311_v5 | Train Models after removing label outliers |
| 6 | Boston311_v6 | Further exploration of label outliers and improving data cleaning function flexibility |
| 7 | Boston311_v7 | Creating Unit Tests for our data clean functions |
| 8 | Boston311_v8 | Creating More Data Cleaning Scenarios and Models |
| 9 | Boston311_v9 | Converting our code to a python package on GitHub |
| 10 | Boston311_v10 | Rewriting all our code as a single Boston311Model class |