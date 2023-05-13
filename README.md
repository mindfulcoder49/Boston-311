# Boston 311 Python Package

This repository contains a Python package for working with the Boston 311 service request data. The package provides functionality for loading and cleaning the data, training machine learning models, and running unit tests.

## Installation

To install the package, you can use pip:

```
pip install git+https://github.com/mindfulcoder49/Boston_311.git
```

## Usage

### Loading the Data

To load the data, you can use the `load_data` module:

```python
data = load_data_from_urls()
```


### Cleaning the Data
To clean the data, you can use the `data_clean` module:

```python
logistic_X, logistic_y = clean_and_split_for_logistic(data, scenario)
linear_X, linear_y = clean_and_split_for_linear(data, scenario)
```

There are two functions: `clean_and_split_for_logistic` and `clean_and_split_for_linear`. Both functions receive a pandas dataframe `myData` and a list of integers `scenario` that specifies which data cleaning steps should be applied to the data.

#### Basic logistic model

No outlier removal. y output is a series of 0 or 1 corresponding to whether a case is Open or Closed, with 0 marking Open and 1 marking closed. X dataframe should contain only dummied columns for the 'subject', 'reason', 'department', 'source', and 'ward_number' columns.

#### Basic linear model

No outlier removal. y output is a series of floats corresponding to the number of hours between case open date and case close date. All open cases are dropped. 


| Model Type | Cleaning Scenario | Description |
| --- | --- | --- |
| Logistic | 0 | No Change from basic |
| Logistic | 1 | Drop any open cases from the last month. |
| Logistic | 2 | Switch the event value for any cases that took longer than a month to close. |
| Logistic | 3 | All records where source is "Employee Generated" or "City Worker App" are removed. |
| Logistic | 4 | All records where survival_time is less than an hour are removed. |
| Logistic | 5 | The type column is one-hot encoded and added to the data. |
| Linear | 0 | No change from basic|
| Linear | 1 | Remove records if the case took more than a month to close. |
| Linear | 2 | Remove records only if the time to close is negative. |
| Linear | 3 | All records where source is "Employee Generated" or "City Worker App" are removed. |
| Linear | 4 | All records where survival_time is less than an hour are removed. |
| Linear | 5 | The type column is one-hot encoded and added to the data. |


### Training Machine Learning Models

To train machine learning models, you can use the `train_models` module:

```python

model = train_logistic_model(data, scenario)
model = train_linear_model(data, scenario)
```

### Running Unit Tests

THe unit_tests.py module contains a function to run unit tests on the data cleaning module

```python
test_data_clean_functions() 
```

## Results

The `results` folder contains Jupyter notebooks used for developing the code in this repository.

###Table of Contents

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
