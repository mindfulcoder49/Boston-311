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

The `clean_and_split_for_logistic` function first creates a copy of the input dataframe. It then converts the `open_dt` and `close_dt` columns to datetime, calculates the `survival_time` as the difference between `closed_dt` and `open_dt`, and creates a binary `event` column that is 1 if the case was closed and 0 otherwise. The `ward_number` column is extracted from the `ward` column. Then, the function applies different scenarios to the data, depending on the integers in the `scenario` list:

- Scenario 1: any open cases from the last month are dropped, considering the date 2023-04-09 as the cutoff date.
- Scenario 2: the `event` value is switched to 0 for any cases that took longer than a month to close.
- Scenario 3: all records where `source` is "Employee Generated" or "City Worker App" are removed.
- Scenario 4: all records where `survival_time` is less than an hour are removed.
- Scenario 5: the `type` column is one-hot encoded and added to the data.

After applying the scenarios, the function drops some columns that are not needed and one-hot encodes the categorical columns listed in `dummy_list`. Finally, the function creates a `X` dataframe with the one-hot encoded data, dropping the `case_enquiry_id`, `event`, and `survival_time` columns, and creates a `y` series with the `event` column.

The `clean_and_split_for_linear` function is very similar to `clean_and_split_for_logistic`, but it also drops cases with missing `survival_time`, creates a `survival_time_hours` column with the `survival_time` in hours, and does not drop the `survival_time` column. Additionally, it does not apply scenario 1 (drop open cases from the last month) or scenario 2 (switch `event` to 0 for cases that took longer than a month to close).


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
