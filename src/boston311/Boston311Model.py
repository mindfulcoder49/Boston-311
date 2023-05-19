from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
import json
import pickle 


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


    def save(self, filepath):
        # Save keras model
        if self.model_type == "linear"  :
            self.model.save(filepath + '/linear_keras_model.h5')
        if self.model_type == "logistic" :
            self.model.save(filepath + '/logistic_keras_model.h5')
        if self.model_type == "cox" :
            with open(filepath + '/cox_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        if self.model_type == "tree" :
            with open(filepath + '/decision_tree.pkl', 'wb') as f:
                pickle.dump(self.model, f)

        # Save other properties
        with open(filepath + '/properties.json', 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'feature_dict': self.feature_dict,
                'train_date_range': self.train_date_range,
                'predict_date_range': self.predict_date_range,
                'scenario': self.scenario,
                'model_type': self.model_type,
            }, f)

    def load(self, json_file, model_file):

        # Load other properties
        with open(json_file, 'r') as f:
            properties = json.load(f)
            self.feature_columns = properties['feature_columns']
            self.feature_dict = properties['feature_dict']
            self.train_date_range = properties['train_date_range']
            self.predict_date_range = properties['predict_date_range']
            self.scenario = properties['scenario']
            self.model_type = properties['model_type']
    
        if self.model_type == "logistic" or self.model_type == "linear" :
            self.model = keras.models.load_model(model_file)
        if self.model_type == "cox" :
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        if self.model_type == "tree" :
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)


    #load_data() - this will use the start_date and end_date. It will return a dataframe
    def load_data(self, train_or_predict='train') :
        start_date, end_date = None, None
        if train_or_predict == 'train' :
            start_date = pd.to_datetime(self.train_date_range['start'])
            end_date = pd.to_datetime(self.train_date_range['end'])
        elif train_or_predict == 'predict' :
            start_date = pd.to_datetime(self.predict_date_range['start'])
            end_date = pd.to_datetime(self.predict_date_range['end'])
        data = self.load_data_from_urls(range(start_date.year, end_date.year+1))

        data['open_dt'] = pd.to_datetime(data['open_dt'])
        data = data[(data['open_dt'] >= start_date) & (data['open_dt'] <= end_date)]
            
        return data 

    
    #enhance_data( data ) - this will enhance the data according to our needs
    def enhance_data(self, data, train_or_predict='train') :
        data['closed_dt'] = pd.to_datetime(data['closed_dt'])
        data['open_dt'] = pd.to_datetime(data['open_dt'])
        data['survival_time'] = data['closed_dt'] - data['open_dt']
        data['event'] = data['closed_dt'].notnull().astype(int)
        data['ward_number'] = data['ward'].str.extract(r'0*(\d+)')

        # initialize a new column with NaN values
        data['survival_time_hours'] = np.nan  

        # create a boolean mask for non-NaN values
        mask = data['survival_time'].notna()  
        data.loc[mask, 'survival_time_hours'] = data.loc[mask, 'survival_time'].apply(lambda x: x.total_seconds() / 3600)

        if train_or_predict == 'predict' :
            #drop closed cases
            data = data[(data['event'] == 0)]

            for key, value in self.scenario.items() :
              if key == 'dropColumnValues' :
                  for column, column_values in value.items() :
                      data = data[~data[column].isin(column_values)]
              if key == 'keepColumnValues' :
                  for column, column_values in value.items() :
                      data = data[data[column].isin(column_values)]

        if self.model_type == 'linear' and train_or_predict == 'train' and 'survivalTimeFill' not in self.scenario :
            #drop open cases
            data = data[(data['event'] == 1)]

        return data
    
    '''
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
        survivalTimeFill - fill survival_time and survival_time_hours as though they were closed on a given date
            value: datestring
            e.g. 2023-05-14
        
    '''
    def clean_data(self, data) :
        
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
            if key == 'eventToZeroforSurvivalTimeGreaterThan' :
                delta = pd.Timedelta(seconds=value)
                data.loc[(data['event'] == 1) & (data['survival_time'] > delta), 'event'] = 0
            if key == 'survivalTimeFill' :
                date = pd.to_datetime(value)
                # create a boolean mask for non-NaN values
                mask = data['survival_time'].isna() 
                data.loc[mask, 'survival_time'] = date - data.loc[mask, 'open_dt']
                data.loc[mask, 'survival_time_hours'] = data.loc[mask, 'survival_time'].apply(lambda x: x.total_seconds() / 3600)

        
        
        #get a list of all columns not in feature_columns or our two labels
        cols_to_drop = data.columns.difference(self.feature_columns + ['event', 'survival_time_hours'])

        data = data.drop(columns=cols_to_drop, axis=1)

        for column in self.feature_columns :
            self.feature_dict[column] = data[column].unique().tolist()
        
        data = pd.get_dummies(data, columns=self.feature_columns)

        return data


    '''
    clean_data_for_prediction( data ) - this will drop any columns not in feature_columns, and one hot encode the training data for prediction with the model by using the feature_columns and feature_dict to ensure the cleaned data is in the correct format for prediction with this model.
    '''

    def clean_data_for_prediction( self, data ) :
        
        cols_to_drop = data.columns.difference(self.feature_columns + ['event', 'survival_time_hours'])

        data = data.drop(columns=cols_to_drop, axis=1)

        data = self.one_hot_encode_with_feature_dict( data )

        return data

    def one_hot_encode_with_feature_dict( self, data ) :
        
        # Loop through each column in the DataFrame
        for column in data.columns:
            # Get the list of allowed values for this column
            allowed = self.feature_dict.get(column, [])
            
            # Loop through each value in the column
            for i, value in data[column].items():
                # Check if the value is in the list of allowed values
                if value not in allowed:
                    # Replace the value with a null value
                    data.at[i, column] = None
        
        fake_records = []
        for col, vals in self.feature_dict.items():
            missing_vals = set(vals) - set(data[col])
            for val in missing_vals:
                # create a dictionary with null values for all columns in your DataFrame
                fake_record = {col: None for col in data.columns}

                # update the dictionary with the non-null value for the current column
                fake_record[col] = val

                # append the fake record to the list of fake records
                fake_records.append(fake_record)

        fake_df = pd.DataFrame(fake_records)

        # Concatenate fake records with original data
        data = pd.concat([data, fake_df], ignore_index=True)

        # Get dummies and drop fake records
        dummies = pd.get_dummies(data, columns=self.feature_dict.keys())
        dummies = dummies.iloc[:-len(fake_df), :]

        return dummies


    '''
    predict() - this will load the data based on the predict_date_range, call clean_data_for_prediction, call split data, use the model to predict the label, then use the id series to join the predictions with the original data, returning a data frame.
    '''
    def predict( self ) :
        data = self.load_data( 'predict' )
        data = self.enhance_data( data, 'predict')
        clean_data = self.clean_data_for_prediction( data )

        if self.model_type == 'linear' :
            X_predict, y_predict = self.split_data( clean_data )
            y_predict = self.model.predict(X_predict)
            data['survival_prediction'] = y_predict
            data['survival_timedelta'] = data['survival_prediction'].apply(lambda x: pd.Timedelta(seconds=(x*3600)))
            data['closed_dt_prediction'] = data['open_dt'] + data['survival_timedelta']
            return data
        elif self.model_type == 'logistic' :
            X_predict, y_predict = self.split_data( clean_data )
            y_predict = self.model.predict(X_predict)
            data['event_prediction'] = y_predict
            return data
        elif self.model_type == 'tree' :
            X_predict, y_predict = self.split_data( clean_data )
            y_predict = self.model.predict(X_predict)
            data['event_prediction'] = y_predict
            return data
        elif self.model_type == "cox" :
            risks = self.model.predict_partial_hazard(clean_data) 
            survival_function = self.model.predict_survival_function(clean_data)
            median_survival_times = self.model.predict_median(clean_data)
            return risks, survival_function, median_survival_times




    '''
    split_data( data ) - this takes data that is ready for training and splits it into an id series, a feature matrix, and a label series
    '''
    def split_data(self, data) :

        X, y = None, None 
        if self.model_type == 'logistic' :
            X = data.drop(['survival_time_hours', 'event'], axis=1) 
            y = data['event']
            
        if self.model_type == 'linear' :
            X = data.drop(['survival_time_hours', 'event'], axis=1) 
            y = data['survival_time_hours']

        if self.model_type == 'tree' :
            X = data.drop(['survival_time_hours', 'event'], axis=1) 
            y = data['event']
        
        return X, y 

    '''
    train_model( X, y ) - this trains the model and returns the model object
    '''
    def train_model( self, X, y=[] ) :
        if self.model_type == 'logistic' :
            self.model = self.train_logistic_model( X, y )

        if self.model_type == 'linear' :
            self.model = self.train_linear_model( X, y )
        if self.model_type == 'cox' :
            self.model = self.train_cox_model(X)
        if self.model_type == 'tree' :
            self.model = self.train_tree_model( X, y )
            
    
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
    
    def train_cox_model(self, data):
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split the data into a training set, a validation set, and a test set
        df_temp, test_df = train_test_split(data, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(df_temp, test_size=0.25, random_state=42)

        # Fit the Cox proportional hazards model
        model = CoxPHFitter()
        model.fit(train_df, duration_col='survival_time_hours', event_col='event')

        # Predict the risk on the validation set and evaluate
        val_duration = val_df.pop('survival_time_hours')
        val_event_observed = val_df.pop('event')
        val_predictions = model.predict_partial_hazard(val_df)
        c_index = concordance_index(val_duration, -val_predictions, val_event_observed)
        print(f"Concordance Index on Validation Set: {c_index}")

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model
    
    def train_tree_model ( self, tree_X, tree_y ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(tree_X, tree_y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Initialize the model
        model = DecisionTreeClassifier(random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Predict the validation set results
        y_val_pred = model.predict(X_val)

        # Calculate the accuracy
        accuracy = accuracy_score(y_val, y_val_pred)

        print('Validation accuracy:', accuracy)

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model

    def run_pipeline( self, data_original=None) :
        data = None
        if data_original == None :
            data = self.load_data()
        else :
            data = data_original.copy()
        data = self.enhance_data(data)
        data = self.clean_data(data)
        if self.model_type in ['linear','logistic','tree'] :
            X, y = self.split_data(data)
            self.train_model( X, y )
        elif self.model_type == 'cox' :
            self.train_model( data )
    

    def load_data_from_urls(self, *args) :
        url_2023 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/e6013a93-1321-4f2a-bf91-8d8a02f1e62f/download/tmpkcy7n55q.csv"
        url_2022 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/81a7b022-f8fc-4da5-80e4-b160058ca207/download/tmph4izx_fb.csv"
        url_2021 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/f53ebccd-bc61-49f9-83db-625f209c95f5/download/tmppgq9965_.csv"
        url_2020 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/6ff6a6fd-3141-4440-a880-6f60a37fe789/download/script_105774672_20210108153400_combine.csv"
        url_2019 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/ea2e4696-4a2d-429c-9807-d02eb92e0222/download/311_service_requests_2019.csv"
        url_2018 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/2be28d90-3a90-4af1-a3f6-f28c1e25880a/download/311_service_requests_2018.csv"
        url_2017 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/30022137-709d-465e-baae-ca155b51927d/download/311_service_requests_2017.csv"
        url_2016 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a/download/311_service_requests_2016.csv"
        url_2015 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/c9509ab4-6f6d-4b97-979a-0cf2a10c922b/download/311_service_requests_2015.csv"
        url_2014 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/bdae89c8-d4ce-40e9-a6e1-a5203953a2e0/download/311_service_requests_2014.csv"
        url_2013 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/407c5cd0-f764-4a41-adf8-054ff535049e/download/311_service_requests_2013.csv"
        url_2012 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/382e10d9-1864-40ba-bef6-4eea3c75463c/download/311_service_requests_2012.csv"
        url_2011 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/94b499d9-712a-4d2a-b790-7ceec5c9c4b1/download/311_service_requests_2011.csv"


        # Get a list of all CSV files in the directory
        files_dict = {
        '2023': url_2023,
        '2022': url_2022,
        '2021': url_2021,
        '2020': url_2020,
        '2019': url_2019,
        '2018': url_2018,
        '2017': url_2017,
        '2016': url_2016,
        '2015': url_2015,
        '2014': url_2014,
        '2013': url_2013,
        '2012': url_2012,
        '2011': url_2011
        }

        all_files = []
        if args != [] :
            for value in args[0] :
                if str(value) in files_dict.keys() :
                    all_files.append(files_dict[str(value)])
        else :
            all_files = files_dict.values 



            

        # Create an empty list to store the dataframes
        dfs = []

        # Loop through the files and load them into dataframes
        for file in all_files:
            df = pd.read_csv(file)
            dfs.append(df)

        #check that the files all have the same number of columns, and the same names
        same_list_num_col = []
        diff_list_num_col = []
        same_list_order_col = []
        diff_list_order_col = []

        for i in range(len(dfs)):

            if dfs[i].shape[1] != dfs[0].shape[1]:
                #print('Error: File', i, 'does not have the same number of columns as File 0')
                diff_list_num_col.append(i)
            else:
                #print('File', i, 'has same number of columns as File 0')
                same_list_num_col.append(i)
            if not dfs[i].columns.equals(dfs[0].columns):
                #print('Error: File', i, 'does not have the same column names and order as File 0')
                diff_list_order_col.append(i)
            else:
                #print('File', i, 'has the same column name and order as File 0')
                same_list_order_col.append(i)

        print("Files with different number of columns from File 0: ", diff_list_num_col)
        print("Files with same number of columns as File 0: ", same_list_num_col)
        print("Files with different column order from File 0: ", diff_list_order_col)
        print("Files with same column order as File 0: ", same_list_order_col)

        # Concatenate the dataframes into a single dataframe
        df_all = pd.concat(dfs, ignore_index=True)

        return df_all        