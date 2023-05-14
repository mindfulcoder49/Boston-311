from .unit_tests import test_data_clean_functions
from .load_data import load_data_from_urls
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np

class Boston311Model: 
    
    '''
    model - our model once trained
    feature_columns - a list of our feature columns
    feature_dict - a dictionary with the keys being the names of our feature columns and the values being lists of all the possible values
    train_date_range - a dict with keys "start" and "end" and datetime values
    predict_date_range - a dict with keys "start" and "end" and datetime values
    scenario - our scenario data, maybe a list, maybe a dict, depending on how we recode our scenarios
    model_type - Our type of model, linear, logistic, etc
    '''
    def __init__(self, **kwargs) :
        self.model = kwargs.get('model', None) 
        self.feature_columns = kwargs.get('feature_columns', [])
        self.feature_dict = kwargs.get('feature_dict', {})
        self.train_date_range = kwargs.get('train_date_range', {'start':'2010-12-31', 'end':'2030-01-01'})
        self.predict_date_range = kwargs.get('predict_date_range', {'start':'', 'end':''})
        self.scenario = kwargs.get('scenario', {})
        self.model_type = kwargs.get('model_type', 'logistic')


    
    #load_data() - this will use the start_date and end_date. It will return a dataframe
    def load_data(self) :
        start_date = pd.to_datetime(self.train_date_range['start'])
        end_date = pd.to_datetime(self.train_date_range['end'])
        data = load_data_from_urls(range(start_date.year, end_date.year+1))

        data['open_dt'] = pd.to_datetime(data['open_dt'])
        data = data[(data['open_dt'] >= start_date) & (data['open_dt'] <= end_date)]
        return data 

    
    #enhance_data( data ) - this will enhance the data according to our needs
    def enhance_data(self, data) :
        data['closed_dt'] = pd.to_datetime(data['closed_dt'])
        data['survival_time'] = data['closed_dt'] - data['open_dt']
        data['event'] = data['closed_dt'].notnull().astype(int)
        data['ward_number'] = data['ward'].str.extract(r'0*(\d+)')

        # initialize a new column with NaN values
        data['survival_time_hours'] = np.nan  

        # create a boolean mask for non-NaN values
        mask = data['survival_time'].notna()  
        data.loc[mask, 'survival_time_hours'] = data.loc[mask, 'survival_time'].apply(lambda x: x.total_seconds() / 3600)
    
    '''
    clean_data() - this will drop any columns not in feature_columns, create the feature_dict, and one-hot encode the training data

    This is also where we begin applying scenarios. The scenarios parameter is a dict. 
    valid keys and values
    All algorithms:
        dropColumnValues 
            value: a dict of column names and lists of values to drop
            e.g. {'source':['City Worker App', 'Employee Generated']}
        keepColumnValues
            value: a dict of column names and lists of values to keep, all others being dropped
            e.g. {'source':['Constituent Call']}
        dropOpen - drop all open cases after a certain date
            value: datestring
            e.g. '2023-05-13'
        survivalTimeMin - drop all closed cases where survival time is less than a given number of seconds
            value: int, a number of seconds
            e.g. 3600
        survivalTimeMax - drop all closed cases where survival time is more than a given number of seconds
            value: int, a number of seconds
            e.g. 2678400

        implement later:
        survivalTimeFill - fill survival_time and survival_time_hours as though they were closed on a given date
            value: datestring
            e.g. 2023-05-14
        
    '''
    def clean_data(self, data) :
        '''
        for key, value in self.scenario.items() :
            if key == 'dropColumnValues' :
                for column, column_values in value.items() :
                    data = data[~data[column].isin(column_values)]
            if key == 'keepColumnValues' :
                for column, column_values in value.items() :
                    data = data[data[column].isin(column_values)]
            if key == 'dropOpen' :
                data = data[(data['event'] == 1) | (data['open_dt'] < pd.to_datetime(value))]
            if key == 'survivalTimeMin' :
                delta = pd.Timedelta(seconds=value)
                data = data[(data['event'] == 0) | (data['survival_time'] >= delta)]
            if key == 'survivalTimeMax' :
                delta = pd.Timedelta(seconds=value)
                data = data[(data['event'] == 0) | (data['survival_time'] <= delta)]
            # implement later
            # if key == 'survivalTimeFill' :
        '''
        
        #get a list of all columns not in feature_columns or our two labels
        cols_to_drop = data.columns.difference(self.feature_columns + ['event', 'survival_time_hours'])

        data.drop(columns=cols_to_drop, inplace=True)

        for column in self.feature_columns :
            self.feature_dict[column] = data[column].unique().tolist()
        
        data = pd.get_dummies(data, columns=self.feature_columns)




    '''
    split_data( data ) - this takes data that is ready for training and splits it into an id series, a feature matrix, and a label series
    '''
    def split_data(self, data) :
        if self.model_type == 'logistic' :
            X = data.drop(['survival_time_hours', 'event'], axis=1) 
            y = data['event']
            
            return X, y
        if self.model_type == 'linear' :
            X = data.drop(['survival_time_hours', 'event'], axis=1) 
            y = data['survival_time_hours']
        
        return X, y 

    '''
    train_model( X, y ) - this trains the model and returns the model object
    '''
    def train_model( self, X, y ) :
        if self.model_type == 'logistic' :
            self.model = self.train_logistic_model( X, y )

        if self.model_type == 'linear' :
            self.model = self.train_linear_model( X, y )
    
    def train_logistic_model ( self, logistic_X, logistic_y ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(logistic_X, logistic_y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Build model
        model = keras.Sequential([
            keras.layers.Dense(units=1, input_shape=(X_train.shape[1],), activation='sigmoid')
        ])

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

        # Train model with early stopping
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)

        print('Test accuracy:', test_acc)

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model
    
    def train_linear_model( self, linear_X, linear_y ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(linear_X) # scale the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, linear_y, test_size=0.2, random_state=42)

        # split the data again to create a validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # define the model architecture
        model = keras.Sequential([
            keras.layers.Dense(units=1, input_dim=X_train.shape[1])
        ])

        # compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # train the model
        # we are adding early stopping based on the validation loss
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stop])

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model

    def run_pipeline( self ) :
        data = self.load_data()
        self.enhance_data(data)
        self.clean_data(data)
        X, y = self.split_data(data)
        self.train_model( X, y )
    
    '''
    clean_data_for_prediction( data ) - this will drop any columns not in feature_columns, and one hot encode the training data for prediction with the model by using the feature_columns and feature_dict to ensure the cleaned data is in the correct format for prediction with this model.

    predict() - this will load the data based on the predict_date_range, call clean_data_for_prediction, call split data, use the model to predict the label, then use the id series to join the predictions with the original data, returning a data frame.
    '''

        