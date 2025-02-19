import pandas as pd
import numpy as np
from math import pi
from math import cos
import json
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pytz import timezone

#refactor code as separate classes for each model type  - linear, logistic, cox, decision tree

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
        self.files_dict = kwargs.get('files_dict', None)
        #add add_columns which will be a dataframe to add to the data
        self.add_columns = kwargs.get('add_columns', None)
        #self.model_type = kwargs.get('model_type', 'logistic')
        #define self.model_type as the class name of self 
        self.model_type = self.__class__.__name__
        self.bin_edges = kwargs.get('bin_edges', None)
        self.bin_labels = kwargs.get('bin_labels', None)



    def save_properties(self, filepath, properties_file):
        # Save other properties
        with open(filepath + '/' + properties_file + '.json', 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'feature_dict': self.feature_dict,
                'train_date_range': self.train_date_range,
                'predict_date_range': self.predict_date_range,
                'scenario': self.scenario,
                'model_type': self.model_type
            }, f)


    def load_properties(self, json_file) :
        # Load other properties
        with open(json_file, 'r') as f:
            properties = json.load(f)
            self.feature_columns = properties['feature_columns']
            self.feature_dict = properties['feature_dict']
            self.train_date_range = properties['train_date_range']
            self.predict_date_range = properties['predict_date_range']
            self.scenario = properties['scenario']


    def get_datestrings(self, days=30) :
        now = datetime.now()
        days_timedelta = timedelta(days=days)
        X_days_ago = now - days_timedelta
        today_datestring = now.strftime("%Y-%m-%d")
        X_days_ago_datestring = X_days_ago.strftime("%Y-%m-%d")
        tomorrow_datestring = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        return today_datestring, tomorrow_datestring, X_days_ago_datestring
    
    #get current datetime in Boston timezone as string
    
    def get_current_datetime_str(self) :
        boston = timezone('US/Eastern')
        now = datetime.now(boston)
        today_datestring = now.strftime("%Y-%m-%d")
        #get time in Boston timezone as string for a filename
        now = datetime.now(boston)
        time_string = now.strftime("%H-%M-%S")
        #define datetime string
        my_datetime = today_datestring + '_' + time_string 
        return my_datetime
    
    #define a function that takes a path to a csv file and a pkl file and checks if the csv file is newer than the pkl file, and if so, loads the csv file into a dataframe and saves it as a pkl file, else loads the pkl file into a dataframe
    def pkl_load_data(self, csv_path, pkl_path):
        if os.path.exists(pkl_path):
            pkl_time = os.path.getmtime(pkl_path)
            csv_time = os.path.getmtime(csv_path)
            if csv_time > pkl_time:
                df = pd.read_csv(csv_path)
                df.to_pickle(pkl_path)
            else:
                df = pd.read_pickle(pkl_path)
        else:
            df = pd.read_csv(csv_path)
            df.to_pickle(pkl_path)
        return df



    #load_data() - this will use the start_date and end_date. It will return a dataframe
    def load_data(self, data=None, train_or_predict='train') :
        start_date, end_date = None, None
        if train_or_predict == 'train' :
            start_date = pd.to_datetime(self.train_date_range['start'])
            end_date = pd.to_datetime(self.train_date_range['end'])
        elif train_or_predict == 'predict' :
            start_date = pd.to_datetime(self.predict_date_range['start'])
            end_date = pd.to_datetime(self.predict_date_range['end'])
        if data is None :
            data = self.load_data_from_urls(range(start_date.year, end_date.year+1))
        else :
            data = data.copy()

        data['open_dt'] = pd.to_datetime(data['open_dt'])
        data = data[(data['open_dt'] >= start_date) & (data['open_dt'] <= end_date)]
            
        return data 

    
    #enhance_data( data ) - this will enhance the data according to our needs
    def enhance_data(self, data, train_or_predict='train') :
        
        data = data.copy()
        data['closed_dt'] = pd.to_datetime(data['closed_dt'])
        data['open_dt'] = pd.to_datetime(data['open_dt'])
        data['survival_time'] = data['closed_dt'] - data['open_dt']
        data['event'] = data['closed_dt'].notnull().astype(int)
        data['ward_number'] = data['ward'].str.extract(r'0*(\d+)')



        #add seasonality value
        #day_of_year = data['open_dt'].dt.dayofyear
        #data['season_cos'] = day_of_year.apply(lambda x: cos((x - 1) * (2. * pi / 365.25)))

        #add day of week
        #weekday = data['open_dt'].dt.weekday
        #data['weekday_cos'] = weekday.apply(lambda x: cos(x * (2. * pi / 7)))

        # initialize a new column with NaN values
        data['survival_time_hours'] = np.nan  

        # create a boolean mask for non-NaN values
        mask = data['survival_time'].notna()  
        data.loc[mask, 'survival_time_hours'] = data.loc[mask, 'survival_time'].apply(lambda x: x.total_seconds() / 3600)

        if train_or_predict == 'predict' :
            #drop closed cases - edit: Actually it would be better to keep all to see if we correctly identify the ones that are closed
            #data = data[(data['event'] == 0)]

            for key, value in self.scenario.items() :
              if key == 'dropColumnValues' :
                  for column, column_values in value.items() :
                      data = data[~data[column].isin(column_values)]
              if key == 'keepColumnValues' :
                  for column, column_values in value.items() :
                      data = data[data[column].isin(column_values)]

        return data
    
    def apply_scenario(self, data) :

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

        return data

    def clean_data(self, data) :

        #get a list of all columns not in feature_columns or our two labels
        cols_to_drop = data.columns.difference(self.feature_columns + ['event', 'survival_time_hours','case_enquiry_id'])


        data = data.drop(columns=cols_to_drop, axis=1)

        self.feature_dict = {}
        for column in self.feature_columns :
            self.feature_dict[column] = data[column].unique().tolist()
        
        data = pd.get_dummies(data, columns=self.feature_columns)


        return data



    
    #clean_data_for_prediction( data ) - this will drop any columns not in feature_columns, and one hot encode the training data for prediction with the model by using the feature_columns and feature_dict to ensure the cleaned data is in the correct format for prediction with this model.
    

    def clean_data_for_prediction( self, data ) :
        print("columns in data before drop:", data.columns)
        
        cols_to_drop = data.columns.difference(self.feature_columns + ['case_enquiry_id', 'event', 'survival_time_hours'])

        print("columns to drop:", cols_to_drop)

        data = data.drop(columns=cols_to_drop, axis=1)

        print("columns in data before ohewfd:", data.columns)

        data = self.one_hot_encode_with_feature_dict( data )

        return data

    def one_hot_encode_with_feature_dict( self, data ) :
        
        # Loop through each column in the DataFrame
        for column in data.columns:
            # Check if the column is case_enquiry_id and skip it
            if column in ['case_enquiry_id', 'event', 'survival_time_hours']:
                continue
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
    
    def load_data_from_urls(self, *args) :
        url_2023 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/e6013a93-1321-4f2a-bf91-8d8a02f1e62f/download/tmpmbmp9j6w.csv"
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

        print("Checking files_dict")
        if self.files_dict is None :
            print("files_dict is None")
            try :
                print("trying to call get311URLs")
                files_dict = self.get311URLs()
                print("files_dict is", files_dict)
            except :
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
        else :
            files_dict = self.files_dict

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

        #print("Files with different number of columns from File 0: ", diff_list_num_col)
        #print("Files with same number of columns as File 0: ", same_list_num_col)
        #print("Files with different column order from File 0: ", diff_list_order_col)
        #print("Files with same column order as File 0: ", same_list_order_col)

        # Concatenate the dataframes into a single dataframe
        df_all = pd.concat(dfs, ignore_index=True)

        return df_all        
        

    def get311URLs(self) :
        print("trying to get csv URLs")

        # specify the URL of the page
        url = "https://data.boston.gov/dataset/311-service-requests"

        # send a GET request to the URL
        response = requests.get(url)

        # parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the current date and time
        now = datetime.now()

        # Extract the year and print it
        current_year = now.year

        URL_dict = {}

        # find all the anchor tags in the HTML
        # and print out the href attribute, which is the URL
        for link in soup.find_all('a'):
            url = link.get('href')
            if url.endswith('.csv'):
                print("Found URL:", url, "for year", current_year)
                URL_dict[str(current_year)] = url
                current_year = current_year - 1 
        return URL_dict
    
        # Function 1: Generate bin_edges using a fixed hour interval
    def generate_time_bins_fixed_interval(self, hour_interval, max_days):
        max_hours = max_days * 24
        # bin_edges = [0] + [1.3 ** i for i in range(1, int(math.log(max_hours, 1.5)) + 1)]
        bin_edges = [i for i in range(0, max_hours + 1, hour_interval)]
        # bin_edges_days = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,21,28,35,42,49,56,63,70,80,90,100,120,160,180,365,712]
        # for i in bin_edges_days :
        #     bin_edges.append(i*24)
        bin_edges.append(1000000)
        return bin_edges

    # Function 2: Generate bin_edges using statistics
    def generate_time_bins_statistics(self, df, num_intervals=60):
        # Sort DataFrame by survival_time_hours
        df = df.sort_values(by='survival_time_hours')
        # Calculate the size for each bin
        bin_size = len(df) // num_intervals
        # Get bin edges
        bin_edges = [df['survival_time_hours'].iloc[i * bin_size] for i in range(num_intervals)]
        bin_edges.append(df['survival_time_hours'].max())  # add the maximum value
        bin_edges = [0] + bin_edges  # add 0 at the beginning
        return bin_edges
    
    def time_format(self, hours):
        if hours == 0:
            return "0"
        seconds = hours * 3600
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        weeks, days = divmod(days, 7)
        years, weeks = divmod(weeks, 52)

        components = [("y", years), ("w", weeks), ("d", days), ("h", hours), ("m", minutes), ("s", seconds)]
        label = "".join([f"{value}{unit}" for unit, value in components if value > 0])
        return label

    def generate_bin_labels(self, bin_edges, overflow_label=None):
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            start_label = self.time_format(bin_edges[i])
            end_label = self.time_format(bin_edges[i + 1])
            if start_label != end_label:
                label = f"{start_label}-{end_label}"
            else:
                label = start_label
            bin_labels.append(label)

        if overflow_label is not None:
            bin_labels[-1] = overflow_label

        return bin_labels