# %% [markdown]
# #Boston 311 Tutorial
# 
# This notebook will run you through the basic usage of this package to train 3 models on the Boston 311 data and use them to predict the outcome of cases from the last 30 days

# %% [markdown]
# ##Install the package from github using pip

# %%
#This library is only needed for the Cox Regression Model, which is not included in this tutorial
#! pip install lifelines

# %%
#pwd()

# %%
! pip install ../

# %% [markdown]
# ##Import the Boston311Model class

# %%
! pip show boston311

# %%
from boston311 import Boston311LogReg, Boston311EventDecTree, Boston311SurvDecTree

# %% [markdown]
# #Get latest file URLS and Current Date Ranges

# %%
! ls .

# %%
import os

#define daily model folder constant
DAILY_MODEL_FOLDER = './daily_models'


# The helper function load_model_from_file is adjusted to load a model 
# based on its type and the provided timestamp.
# The main loop iterates through each folder in DAILY_MODEL_FOLDER.
# For each folder, it checks for model files (.pkl or .h5).
# If a model file is found, it extracts the timestamp and model type 
# from the filename and uses the helper function to load the model.
# The loaded model is added to the daily_model_dict with the key being 
# the model's filename without the extension.


def load_model_from_file(model_type, folder_path, timestamp):
    """Load a model based on its type from a given folder."""
    if model_type == 'Boston311EventDecTree':
        model_instance = Boston311EventDecTree()
        model_file = f'{timestamp}_{model_type}.pkl'
    elif model_type == 'Boston311LogReg':
        model_instance = Boston311LogReg()
        model_file = f'{timestamp}_{model_type}.h5'
    elif model_type == 'Boston311SurvDecTree':
        model_instance = Boston311SurvDecTree()
        model_file = f'{timestamp}_{model_type}.pkl'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    properties_file = f'{timestamp}_{model_type}.json'
    model_instance.load(os.path.join(folder_path, properties_file), os.path.join(folder_path, model_file))
    
    return model_instance

daily_model_dict = {}

for folder in os.listdir(DAILY_MODEL_FOLDER):
    folder_path = os.path.join(DAILY_MODEL_FOLDER, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.count('_') == 2 and any(ext in file for ext in ['.pkl', '.h5']):
                parts = file.rsplit('.', 1)[0].split('_')
                timestamp = f"{parts[0]}_{parts[1]}"
                model_type = parts[2]
                try:
                    daily_model_dict[f'{timestamp}_{model_type}'] = load_model_from_file(model_type, folder_path, timestamp)
                except ValueError:
                    # Skip files with unknown model types
                    continue

daily_model_dict



# %%
from datetime import datetime, timedelta
now = datetime.now()
today_datestring = now.strftime("%Y-%m-%d")

# %%
import pandas as pd

#define an empt pandas dataframe ml_model_df
ml_model_df = pd.DataFrame(columns=['ml_model_name', 'ml_model_type', 'ml_model_date'])
all_model_cases = pd.DataFrame()
all_model_predictions = pd.DataFrame()


ml_model_df

# %%



#foreach model in the daily_model_dict set the predict_dat_range to the last 30 days and then call the predict method and save the results to a csv file
for model_name, model in daily_model_dict.items():

    print(f"Processing model: {model_name}")

    print(ml_model_df)
    model.predict_date_range = {'start': '2023-08-09', 'end': today_datestring}



    #get file creation date for the .json file in the folder with the model_name
    #use os.path.getctime to get the creation time of the .json file in the folder with the model_name
    #convert the creation time to a datetime object
    #convert the datetime object to a string in the format of %Y-%m-%d
    #add to ml_model_df dataframe with  columns, ml_model_name, ml_model_type,ml_model_id, ml_model_date
    ml_model_datetime = os.path.getctime(os.path.join(DAILY_MODEL_FOLDER, model.__class__.__name__, model_name + '.json'))
    ml_model_date = datetime.fromtimestamp(ml_model_datetime).strftime('%Y-%m-%d')
    

    ml_model_df = pd.concat([ml_model_df, pd.DataFrame([{'ml_model_name': model_name, 
                                    'ml_model_type': model.__class__.__name__,
                                    'ml_model_date': ml_model_date}])], ignore_index=True)
    
    print(ml_model_df)

    model_prediction = model.predict()

    #check if the model_prediction dataframe contains an event_prediction column
    if 'event_prediction' in model_prediction.columns:
    #get new dataframe with just the event_prediction column from the model_prediction dataframe
        model_prediction_event = model_prediction[['event_prediction','case_enquiry_id']].copy()
        model_prediction_event.rename(columns={'event_prediction': 'prediction'}, inplace=True)
        #remove model_prediction event_prediction column
        model_cases = model_prediction.drop('event_prediction', axis=1).copy()
    elif 'survival_prediction' in model_prediction.columns:
        model_prediction_event = model_prediction[['survival_prediction','case_enquiry_id']].copy()
        model_prediction_event.rename(columns={'survival_prediction': 'prediction'}, inplace=True)
        #remove model_prediction survival_prediction column
        model_cases = model_prediction.drop('survival_prediction', axis=1).copy()

    model_prediction_event['ml_model_name'] = model_name
    #add today's date to the dataframe
    model_prediction_event['prediction_date'] = today_datestring
    #rename the event_prediction column to prediction
    
    #remove geom column in model_cases
    model_cases = model_cases.drop('geom_4326', axis=1).copy()

    # Add the model_cases dataframe to the all_model_cases dataframe
    all_model_cases = pd.concat([all_model_cases, model_cases], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Add the model_prediction_event dataframe to the all_model_predictions dataframe
    all_model_predictions = pd.concat([all_model_predictions, model_prediction_event], ignore_index=True)

    

    



# %%
#count rows in prediction dataframe
print(f"Number of rows in all_model_predictions: {len(all_model_predictions)}")

# %%
# Assuming the dataframe with all case data is named all_cases
closed_case_ids = all_model_cases[all_model_cases['case_status'] == 'Closed']['case_enquiry_id'].unique()

# Drop rows from all_model_predictions where case_enquiry_id matches those in closed_case_ids
all_model_predictions = all_model_predictions[~all_model_predictions['case_enquiry_id'].isin(closed_case_ids)]


# %%
#count rows in prediction dataframe
print(f"Number of rows in all_model_predictions: {len(all_model_predictions)}")

# %% [markdown]
# ## Save the prediction data
# 

# %%
#get current datetime in Boston timezone as string
from datetime import datetime
from pytz import timezone
import pytz
boston = timezone('US/Eastern')
now = datetime.now(boston)
today_datestring = now.strftime("%Y-%m-%d")
#get time in Boston timezone as string for a filename
now = datetime.now(boston)
time_string = now.strftime("%H-%M-%S")
#define datetime string
my_datetime = today_datestring + '_' + time_string 

my_datetime

# %%
all_model_cases.to_csv(my_datetime+'_311_cases.csv', index=False)


# %%

all_model_predictions.to_csv(my_datetime+'_311_predictions.csv', index=False)

# %%

ml_model_df.to_csv(my_datetime+'_311_ml_models.csv', index=False)

# %%
#create datetime _manifest.txt file with one filename per line
with open(my_datetime+'_manifest.txt', 'w') as f:
    f.write(my_datetime+'_311_cases.csv\n')
    f.write(my_datetime+'_311_predictions.csv\n')
    f.write(my_datetime+'_311_ml_models.csv\n')

# %%
#create an export folder
EXPORT_FOLDER = '~/Documents/BODC-DEI-site/database/seeders'
#copy the csv files to the export folder
!cp {my_datetime}_311_cases.csv {EXPORT_FOLDER}
!cp {my_datetime}_311_predictions.csv {EXPORT_FOLDER}
!cp {my_datetime}_311_ml_models.csv {EXPORT_FOLDER}
!cp {my_datetime}_manifest.txt {EXPORT_FOLDER}



# %% [markdown]
# ** Copy the files to the production server **

# %%
import os

# Define constants for servers
PROD_USER = 'u353344964'
PROD_HOSTNAME = '195.179.236.61'
PORT_NUMBER = 65002
PROD_EXPORT_FOLDER = '/home/u353344964/domains/bodc-dei.org/laravel/database/seeders'
STAGE_EXPORT_FOLDER = '/home/u353344964/domains/bodc-dei.org/stagelaravel/database/seeders'

def scp_to_server(filename, user=PROD_USER, hostname=PROD_HOSTNAME, port=PORT_NUMBER, export_folder=PROD_EXPORT_FOLDER):
    """Copy a file to the server using scp."""
    command = f"scp -P {port} {filename} {user}@{hostname}:{export_folder}"
    print(f"Executing: {command}")
    os.system(command)

# Use the function to scp files
files_to_copy = [
    f"{my_datetime}_311_cases.csv",
    f"{my_datetime}_311_predictions.csv",
    f"{my_datetime}_311_ml_models.csv",
    f"{my_datetime}_manifest.txt"
]

# Control where to copy
copy_to_prod = True
copy_to_stage = True

for file in files_to_copy:
    if copy_to_prod:
        scp_to_server(file, export_folder=PROD_EXPORT_FOLDER)
    if copy_to_stage:
        scp_to_server(file, export_folder=STAGE_EXPORT_FOLDER)



