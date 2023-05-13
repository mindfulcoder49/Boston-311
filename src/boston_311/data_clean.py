import pandas as pd

def clean_and_split_for_logistic(myData, scenario) :

  data = myData.copy()
  # Convert the 'open_dt' and 'close_dt' columns to datetime
  data['open_dt'] = pd.to_datetime(data['open_dt'])
  data['closed_dt'] = pd.to_datetime(data['closed_dt'])
  data['survival_time'] = data['closed_dt'] - data['open_dt']
  data['event'] = data['closed_dt'].notnull().astype(int)
  data['ward_number'] = data['ward'].str.extract(r'0*(\d+)')

  #this is a comment

  cols_to_keep = ['case_enquiry_id', 'survival_time', 'event', 'subject', 'reason', 'department', 'source', 'ward_number']
  cols_to_drop = [
 'open_dt',
 'target_dt',
 'closed_dt',
 'ontime',
 'case_status',
 'closure_reason',
 'case_title',
 'type',
 'queue',
 'submittedphoto',
 'closedphoto',
 'location',
 'fire_district',
 'pwd_district',
 'city_council_district',
 'police_district',
 'neighborhood',
 'neighborhood_services_district',
 'ward',
 'precinct',
 'location_street_name',
 'location_zipcode',
 'latitude',
 'longitude']

  #scenarios
  #scenario 0: no outlier adjustments
  
  #scenario 1: drop any open cases from the last month
  if 1 in scenario :
    # Convert the date string to a pandas Timestamp object
    cutoff_date = pd.Timestamp('2023-04-09')

    # Filter the DataFrame to include only rows where event is 1 or open_dt is before the cutoff date
    data = data[(data['event'] == 1) | (data['open_dt'] < cutoff_date)]

    #switch the event value for any cases that took longer than a month to close

  #scenario 2: switch the event value for any cases that took longer than a month to close
  if 2 in scenario :
    delta = pd.Timedelta(seconds=2678400)
    data.loc[(data['event'] == 1) & (data['survival_time'] > delta), 'event'] = 0

  #scenario 3: Remove all records where source is "Employee Generated" or "City Worker App"
  if 3 in scenario :
    data = data[~data['source'].isin(["Employee Generated", "City Worker App"])]

  #scenario 4: Remove all records where survival time is less than an hour
  if 4 in scenario :
    delta = pd.Timedelta(seconds=3600)
    data = data[(data['event'] == 0) | (data['survival_time'] < delta)]


  dummy_list = ['subject', 'reason', 'department', 'source', 'ward_number']

  #scenario 5: Add type as a one hot encoded categorical variable
  if 5 in scenario :
    dummy_list.append('type')
    cols_to_drop.remove('type')



  data = data.drop(cols_to_drop, axis=1)

  data = pd.get_dummies(data, columns=dummy_list)



  #fix this line to also drop the case_enquiry_id
  X = data.drop(['case_enquiry_id','event', 'survival_time'], axis=1)
  y = data['event']

  return X, y

def clean_and_split_for_linear(myData, scenario) :

  data = myData.copy()
  # Convert the 'open_dt' and 'close_dt' columns to datetime
  data['open_dt'] = pd.to_datetime(data['open_dt'])
  data['closed_dt'] = pd.to_datetime(data['closed_dt'])
  data['survival_time'] = data['closed_dt'] - data['open_dt']
  data['event'] = data['closed_dt'].notnull().astype(int)
  data['ward_number'] = data['ward'].str.extract(r'0*(\d+)')

  cols_to_keep = ['case_enquiry_id', 'survival_time', 'event', 'subject', 'reason', 'department', 'source', 'ward_number']
  cols_to_drop = [
 'open_dt',
 'target_dt',
 'closed_dt',
 'ontime',
 'case_status',
 'closure_reason',
 'case_title',
 'type',
 'queue',
 'submittedphoto',
 'closedphoto',
 'location',
 'fire_district',
 'pwd_district',
 'city_council_district',
 'police_district',
 'neighborhood',
 'neighborhood_services_district',
 'ward',
 'precinct',
 'location_street_name',
 'location_zipcode',
 'latitude',
 'longitude']

  #scenario 3: Remove all records where source is "Employee Generated" or "City Worker App"
  if 3 in scenario :
    data = data[~data['source'].isin(["Employee Generated", "City Worker App"])]

  dummy_list = ['subject', 'reason', 'department', 'source', 'ward_number']
  
  #scenario 5: Add type as a one hot encoded categorical variable
  if 5 in scenario :
    dummy_list.append('type')
    cols_to_drop.remove('type')



  data = data.drop(cols_to_drop, axis=1)

  data = pd.get_dummies(data, columns=dummy_list)

  data_survival_mask = data["survival_time"].notnull()
  clean_data = data[data_survival_mask].copy()
  clean_data['survival_time_hours'] = clean_data['survival_time'].apply(lambda x: x.total_seconds()/3600)

  #add scenarios
  #scenario 0: no outlier adjustments

  #scenario 1: remove records if the case took more than a month to close
  if 1 in scenario :
    clean_data = clean_data[(clean_data['survival_time_hours'] <= 744)]
  
  #scenario 2: remove records just if the time to close is negative 
  if 2 in scenario :
    clean_data = clean_data[(clean_data['survival_time_hours'] >= 0)]



  #scenario 4: Remove all records where survival time is less than an hour
  if 4 in scenario :
    clean_data = clean_data[(clean_data['survival_time_hours'] >= 1)]

  #scenario 5: Add type as a one hot encoded categorical variable

  #fix this line to also drop the case_enquiry_id
  X = clean_data.drop(['case_enquiry_id','survival_time_hours', 'survival_time', 'event'], axis=1) 
  y = clean_data['survival_time_hours']
  
  return X, y