import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from .Boston311Model import Boston311Model

def test_data_clean_functions() :
    #set up the test data
    test_data_2022 = pd.DataFrame({'case_enquiry_id': [101004125189,
                        101004161747,
                        101004149944,
                        101004113302,
                        101004122704,
                        101004122479,
                        101004113310,
                        101004113311,
                        101004113328,
                        101004113550],
    'case_status': ['Open',
                    'Closed',
                    'Open',
                    'Closed',
                    'Open',
                    'Open',
                    'Closed',
                    'Closed',
                    'Open',
                    'Closed'],
    'case_title': ['Illegal Rooming House',
                'PublicWorks: Complaint',
                'Space Savers',
                'Parking Enforcement',
                'DISPATCHED Heat - Excessive  Insufficient',
                'Generic Noise Disturbance',
                'Parking Enforcement',
                'General Lighting Request',
                'Loud Parties/Music/People',
                'Requests for Street Cleaning'],
    'city_council_district': ['4', ' ', '4', '2', '7', '8', '3', '6', '1', '2'],
    'closed_dt': [np.nan,
                '2021-02-02 11:45:47',
                np.nan,
                '2022-01-03 00:13:17',
                np.nan,
                np.nan,
                '2022-01-03 00:13:02',
                '2022-04-02 13:01:14',
                np.nan,
                '2022-05-03 05:59:20'],
    'closedphoto': [np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    'https://spot-boston-res.cloudinary.com/image/upload/v1641207557/boston/production/o0vkrv9zckukp8httr7g.jpg'],
    'closure_reason': [' ',
                    'Case Closed Case Noted    ',
                    ' ',
                    'Case Closed. Closed date : 2022-01-03 00:13:17.393 Case '
                    'Resolved CLEAR ',
                    ' ',
                    ' ',
                    'Case Closed. Closed date : 2022-01-03 00:13:02.72 Case '
                    'Resolved CLEAR ',
                    'Case Closed. Closed date : Sat Apr 02 13:01:14 EDT 2022 '
                    'Noted ',
                    ' ',
                    'Case Closed. Closed date : Mon Jan 03 05:59:20 EST 2022 '
                    'Noted 3 bags of trash collected at intersection of '
                    'Dartmouth and Warren at 5:56 a.m. on Monday 1/3/22. We '
                    'will return on next scheduled trash day. '],
    'department': ['ISD',
                'PWDx',
                'PWDx',
                'BTDT',
                'ISD',
                'INFO',
                'BTDT',
                'PWDx',
                'INFO',
                'PWDx'],
    'fire_district': ['8', ' ', '8', '6', '9', '3', '8', '9', '3', '4'],
    'latitude': [42.2896,
                42.3594,
                42.2876,
                42.3594,
                42.311,
                42.3657,
                42.291,
                42.3594,
                42.3669,
                42.3594],
    'location': ['27 Lithgow St  Dorchester  MA  02124',
                ' ',
                '492 Harvard St  Dorchester  MA  02124',
                'INTERSECTION of Seaport Blvd & Sleeper St  Boston  MA  ',
                '15 Crawford St  Dorchester  MA  02121',
                '50-150 Causeway St  Boston  MA  02114',
                '16 Frost Ave  Dorchester  MA  02122',
                'INTERSECTION of Boylston St & Moraine St  Jamaica Plain  MA  ',
                '194 Salem St  Boston  MA  02113',
                'INTERSECTION of Warren Ave & Dartmouth St  Boston  MA  '],
    'location_street_name': ['27 Lithgow St',
                            np.nan,
                            '492 Harvard St',
                            'INTERSECTION Seaport Blvd & Sleeper St',
                            '15 Crawford St',
                            '50-150 Causeway St',
                            '16 Frost Ave',
                            'INTERSECTION Boylston St & Moraine St',
                            '194 Salem St',
                            'INTERSECTION Warren Ave & Dartmouth St'],
    'location_zipcode': [2124.0,
                        np.nan,
                        2124.0,
                        np.nan,
                        2121.0,
                        2114.0,
                        2122.0,
                        np.nan,
                        2113.0,
                        np.nan],
    'longitude': [-71.0701,
                -71.0587,
                -71.0936,
                -71.0587,
                -71.0841,
                -71.0617,
                -71.0503,
                -71.0587,
                -71.0546,
                -71.0587],
    'neighborhood': ['Dorchester',
                    ' ',
                    'Greater Mattapan',
                    'South Boston / South Boston Waterfront',
                    'Roxbury',
                    'Boston',
                    'Dorchester',
                    'Jamaica Plain',
                    'Downtown / Financial District',
                    'South End'],
    'neighborhood_services_district': ['8',
                                    ' ',
                                    '9',
                                    '5',
                                    '13',
                                    '3',
                                    '7',
                                    '11',
                                    '3',
                                    '6'],
    'ontime': ['OVERDUE',
            'ONTIME',
            'ONTIME',
            'ONTIME',
            'OVERDUE',
            'ONTIME',
            'ONTIME',
            'OVERDUE',
            'ONTIME',
            'ONTIME'],
    'open_dt': ['2023-05-09 12:59:00',
                '2022-02-02 11:42:49',
                '2022-01-28 19:36:00',
                '2022-01-01 00:36:24',
                '2022-01-11 09:47:00',
                '2022-01-10 21:49:00',
                '2022-01-01 01:13:52',
                '2022-01-01 01:14:39',
                '2022-01-01 03:08:00',
                '2022-01-01 13:51:00'],
    'police_district': ['C11',
                        ' ',
                        'B3',
                        'C6',
                        'B2',
                        'A1',
                        'C11',
                        'E13',
                        'A1',
                        'D4'],
    'precinct': ['1706',
                ' ',
                '1411',
                '0601',
                '1202',
                ' ',
                '1607',
                '1903',
                '0302',
                '0401'],
    'pwd_district': ['07', ' ', '07', '05', '10B', '1B', '07', '02', '1B', '1C'],
    'queue': ['ISD_Housing (INTERNAL)',
            'PWDx_General Comments',
            'PWDx_Space Saver Removal',
            'BTDT_Parking Enforcement',
            'ISD_Housing (INTERNAL)',
            'INFO01_GenericeFormforOtherServiceRequestTypes',
            'BTDT_Parking Enforcement',
            'PWDx_Street Light_General Lighting Request',
            'INFO01_GenericeFormforOtherServiceRequestTypes',
            'PWDx_Missed Trash\\Recycling\\Yard Waste\\Bulk Item'],
    'reason': ['Building',
            'Employee & General Comments',
            'Sanitation',
            'Enforcement & Abandoned Vehicles',
            'Housing',
            'Generic Noise Disturbance',
            'Enforcement & Abandoned Vehicles',
            'Street Lights',
            'Noise Disturbance',
            'Street Cleaning'],
    'source': ['Constituent Call',
            'Constituent Call',
            'Constituent Call',
            'Citizens Connect App',
            'Constituent Call',
            'Constituent Call',
            'Citizens Connect App',
            'City Worker App',
            'Constituent Call',
            'Citizens Connect App'],
    'subject': ['Inspectional Services',
                "Mayor's 24 Hour Hotline",
                'Public Works Department',
                'Transportation - Traffic Division',
                'Inspectional Services',
                "Mayor's 24 Hour Hotline",
                'Transportation - Traffic Division',
                'Public Works Department',
                'Boston Police Department',
                'Public Works Department'],
    'submittedphoto': [np.nan,
                    np.nan,
                    np.nan,
                    'https://311.boston.gov/media/boston/report/photos/61cfe84b05bbcf180c293ece/photo_20220101_003547.jpg',
                    np.nan,
                    np.nan,
                    'https://311.boston.gov/media/boston/report/photos/61cff11805bbcf180c2944b1/report.jpg',
                    np.nan,
                    np.nan,
                    'https://311.boston.gov/media/boston/report/photos/61d0a2af05bbcf180c2993e3/report.jpg'],
    'target_dt': ['2022-01-20 12:59:39',
                '2022-02-16 11:42:49',
                np.nan,
                '2022-01-04 08:30:00',
                '2022-02-10 09:47:22',
                np.nan,
                '2022-01-04 08:30:00',
                '2022-02-15 01:14:45',
                np.nan,
                '2022-01-04 08:30:00'],
    'type': ['Illegal Rooming House',
            'General Comments For a Program or Policy',
            'Space Savers',
            'Parking Enforcement',
            'Heat - Excessive  Insufficient',
            'Undefined Noise Disturbance',
            'Parking Enforcement',
            'General Lighting Request',
            'Loud Parties/Music/People',
            'Requests for Street Cleaning'],
    'ward': ['Ward 17',
            ' ',
            'Ward 14',
            '6',
            'Ward 12',
            '03',
            'Ward 16',
            '19',
            'Ward 3',
            '4']})

    #define the expected output
    tlogistic_test_X_0 = pd.DataFrame({'department_BTDT': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'department_INFO': [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    'department_ISD': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'department_PWDx': [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    'reason_Building': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'reason_Employee & General Comments': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'reason_Enforcement & Abandoned Vehicles': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'reason_Generic Noise Disturbance': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'reason_Housing': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'reason_Noise Disturbance': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'reason_Sanitation': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'reason_Street Cleaning': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'reason_Street Lights': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'source_Citizens Connect App': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    'source_City Worker App': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'source_Constituent Call': [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    'subject_Boston Police Department': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'subject_Inspectional Services': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "subject_Mayor's 24 Hour Hotline": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'subject_Public Works Department': [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    'subject_Transportation - Traffic Division': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'ward_number_12': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'ward_number_14': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'ward_number_16': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'ward_number_17': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ward_number_19': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'ward_number_3': [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    'ward_number_4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'ward_number_6': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]}
    )
    tlogistic_test_X_1 = pd.DataFrame({'department_BTDT': [0, 0, 1, 0, 0, 1, 0, 0, 0],
    'department_INFO': [0, 0, 0, 0, 1, 0, 0, 1, 0],
    'department_ISD': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'department_PWDx': [1, 1, 0, 0, 0, 0, 1, 0, 1],
    'reason_Employee & General Comments': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    'reason_Enforcement & Abandoned Vehicles': [0, 0, 1, 0, 0, 1, 0, 0, 0],
    'reason_Generic Noise Disturbance': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'reason_Housing': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'reason_Noise Disturbance': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'reason_Sanitation': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    'reason_Street Cleaning': [0, 0, 0, 0, 0, 0, 0, 0, 1],
    'reason_Street Lights': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'source_Citizens Connect App': [0, 0, 1, 0, 0, 1, 0, 0, 1],
    'source_City Worker App': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'source_Constituent Call': [1, 1, 0, 1, 1, 0, 0, 1, 0],
    'subject_Boston Police Department': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'subject_Inspectional Services': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "subject_Mayor's 24 Hour Hotline": [1, 0, 0, 0, 1, 0, 0, 0, 0],
    'subject_Public Works Department': [0, 1, 0, 0, 0, 0, 1, 0, 1],
    'subject_Transportation - Traffic Division': [0, 0, 1, 0, 0, 1, 0, 0, 0],
    'ward_number_12': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'ward_number_14': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    'ward_number_16': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'ward_number_19': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'ward_number_3': [0, 0, 0, 0, 1, 0, 0, 1, 0],
    'ward_number_4': [0, 0, 0, 0, 0, 0, 0, 0, 1],
    'ward_number_6': [0, 0, 1, 0, 0, 0, 0, 0, 0]}
    )
    tlinear_test_X_0 = pd.DataFrame({'department_BTDT': [0, 1, 1, 0, 0],
    'department_INFO': [0, 0, 0, 0, 0],
    'department_ISD': [0, 0, 0, 0, 0],
    'department_PWDx': [1, 0, 0, 1, 1],
    'reason_Building': [0, 0, 0, 0, 0],
    'reason_Employee & General Comments': [1, 0, 0, 0, 0],
    'reason_Enforcement & Abandoned Vehicles': [0, 1, 1, 0, 0],
    'reason_Generic Noise Disturbance': [0, 0, 0, 0, 0],
    'reason_Housing': [0, 0, 0, 0, 0],
    'reason_Noise Disturbance': [0, 0, 0, 0, 0],
    'reason_Sanitation': [0, 0, 0, 0, 0],
    'reason_Street Cleaning': [0, 0, 0, 0, 1],
    'reason_Street Lights': [0, 0, 0, 1, 0],
    'source_Citizens Connect App': [0, 1, 1, 0, 1],
    'source_City Worker App': [0, 0, 0, 1, 0],
    'source_Constituent Call': [1, 0, 0, 0, 0],
    'subject_Boston Police Department': [0, 0, 0, 0, 0],
    'subject_Inspectional Services': [0, 0, 0, 0, 0],
    "subject_Mayor's 24 Hour Hotline": [1, 0, 0, 0, 0],
    'subject_Public Works Department': [0, 0, 0, 1, 1],
    'subject_Transportation - Traffic Division': [0, 1, 1, 0, 0],
    'ward_number_12': [0, 0, 0, 0, 0],
    'ward_number_14': [0, 0, 0, 0, 0],
    'ward_number_16': [0, 0, 1, 0, 0],
    'ward_number_17': [0, 0, 0, 0, 0],
    'ward_number_19': [0, 0, 0, 1, 0],
    'ward_number_3': [0, 0, 0, 0, 0],
    'ward_number_4': [0, 0, 0, 0, 1],
    'ward_number_6': [0, 1, 0, 0, 0]}
    )
    tlinear_test_X_1 = pd.DataFrame({'department_BTDT': [1, 1],
    'department_INFO': [0, 0],
    'department_ISD': [0, 0],
    'department_PWDx': [0, 0],
    'reason_Building': [0, 0],
    'reason_Employee & General Comments': [0, 0],
    'reason_Enforcement & Abandoned Vehicles': [1, 1],
    'reason_Generic Noise Disturbance': [0, 0],
    'reason_Housing': [0, 0],
    'reason_Noise Disturbance': [0, 0],
    'reason_Sanitation': [0, 0],
    'reason_Street Cleaning': [0, 0],
    'reason_Street Lights': [0, 0],
    'source_Citizens Connect App': [1, 1],
    'source_City Worker App': [0, 0],
    'source_Constituent Call': [0, 0],
    'subject_Boston Police Department': [0, 0],
    'subject_Inspectional Services': [0, 0],
    "subject_Mayor's 24 Hour Hotline": [0, 0],
    'subject_Public Works Department': [0, 0],
    'subject_Transportation - Traffic Division': [1, 1],
    'ward_number_12': [0, 0],
    'ward_number_14': [0, 0],
    'ward_number_16': [0, 1],
    'ward_number_17': [0, 0],
    'ward_number_19': [0, 0],
    'ward_number_3': [0, 0],
    'ward_number_4': [0, 0],
    'ward_number_6': [1, 0]}
    )
    tlinear_test_X_2 = pd.DataFrame({'department_BTDT': [1, 1, 0, 0],
    'department_INFO': [0, 0, 0, 0],
    'department_ISD': [0, 0, 0, 0],
    'department_PWDx': [0, 0, 1, 1],
    'reason_Building': [0, 0, 0, 0],
    'reason_Employee & General Comments': [0, 0, 0, 0],
    'reason_Enforcement & Abandoned Vehicles': [1, 1, 0, 0],
    'reason_Generic Noise Disturbance': [0, 0, 0, 0],
    'reason_Housing': [0, 0, 0, 0],
    'reason_Noise Disturbance': [0, 0, 0, 0],
    'reason_Sanitation': [0, 0, 0, 0],
    'reason_Street Cleaning': [0, 0, 0, 1],
    'reason_Street Lights': [0, 0, 1, 0],
    'source_Citizens Connect App': [1, 1, 0, 1],
    'source_City Worker App': [0, 0, 1, 0],
    'source_Constituent Call': [0, 0, 0, 0],
    'subject_Boston Police Department': [0, 0, 0, 0],
    'subject_Inspectional Services': [0, 0, 0, 0],
    "subject_Mayor's 24 Hour Hotline": [0, 0, 0, 0],
    'subject_Public Works Department': [0, 0, 1, 1],
    'subject_Transportation - Traffic Division': [1, 1, 0, 0],
    'ward_number_12': [0, 0, 0, 0],
    'ward_number_14': [0, 0, 0, 0],
    'ward_number_16': [0, 1, 0, 0],
    'ward_number_17': [0, 0, 0, 0],
    'ward_number_19': [0, 0, 1, 0],
    'ward_number_3': [0, 0, 0, 0],
    'ward_number_4': [0, 0, 0, 1],
    'ward_number_6': [1, 0, 0, 0]}
    )
    tlogistic_test_y_0 = pd.Series({0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1}
    )
    tlogistic_test_y_1 = pd.Series({1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0}
    )
    tlinear_test_y_0 = pd.Series({1: -8759.950555555555,
    3: 47.61472222222222,
    6: 46.986111111111114,
    7: 2195.776388888889,
    9: 2920.1388888888887}
    )
    tlinear_test_y_1 = pd.Series({3: 47.61472222222222, 6: 46.986111111111114}
    )
    tlinear_test_y_2 = pd.Series({3: 47.61472222222222,
    6: 46.986111111111114,
    7: 2195.776388888889,
    9: 2920.1388888888887}
    )

    #create Boston311Model objects
    


    #call the function with the test data
    '''
    logistic_test_X_0, logistic_test_y_0 = clean_and_split_for_logistic(test_data_2022, [0])
    logistic_test_X_1, logistic_test_y_1 = clean_and_split_for_logistic(test_data_2022, [1, 2])

    linear_test_X_0, linear_test_y_0 = clean_and_split_for_linear(test_data_2022, [0])
    linear_test_X_1, linear_test_y_1 = clean_and_split_for_linear(test_data_2022, [1, 2])
    linear_test_X_2, linear_test_y_2 = clean_and_split_for_linear(test_data_2022, [2])
    '''

    myBoston311Model_0 = Boston311Model(feature_columns=['subject', 'reason', 'department', 'source', 'ward_number' ],
                                      model_type="logistic",
                                      scenario={})
    test_data_enhanced_0 = myBoston311Model_0.enhance_data(test_data_2022)
    test_data_cleaned_0 = myBoston311Model_0.clean_data(test_data_enhanced_0)
    logistic_test_X_0, logistic_test_y_0 = myBoston311Model_0.split_data(test_data_cleaned_0)

    myBoston311Model_1 = Boston311Model(feature_columns=['subject', 'reason', 'department', 'source', 'ward_number' ],
                                      model_type="logistic",
                                      scenario={'dropOpen':'2023-04-09',
                                      'eventToZeroforSurvivalTimeGreaterThan': 2678400})
    test_data_enhanced_1 = myBoston311Model_1.enhance_data(test_data_2022)
    test_data_cleaned_1 = myBoston311Model_1.clean_data(test_data_enhanced_1)
    logistic_test_X_1, logistic_test_y_1 = myBoston311Model_1.split_data(test_data_cleaned_1)

    myBoston311Model_lin0 = Boston311Model(feature_columns=['subject', 'reason', 'department', 'source', 'ward_number' ],
                                      model_type="linear",
                                      scenario={'dropOpen':'2021-12-31'})
    test_data_enhanced_lin0 = myBoston311Model_lin0.enhance_data(test_data_2022)
    test_data_cleaned_lin0 = myBoston311Model_lin0.clean_data(test_data_enhanced_lin0)
    linear_test_X_0, linear_test_y_0 = myBoston311Model_lin0.split_data(test_data_cleaned_lin0)

    myBoston311Model_lin1 = Boston311Model(feature_columns=['subject', 'reason', 'department', 'source', 'ward_number' ],
                                      model_type="linear",
                                      scenario={'dropOpen':'2021-12-31',
                                                'survivalTimeMax':2678400,
                                                'survivalTimeMin':0})
    test_data_enhanced_lin1 = myBoston311Model_lin1.enhance_data(test_data_2022)
    test_data_cleaned_lin1 = myBoston311Model_lin1.clean_data(test_data_enhanced_lin1)
    linear_test_X_1, linear_test_y_1 = myBoston311Model_lin1.split_data(test_data_cleaned_lin1)


    myBoston311Model_lin2 = Boston311Model(feature_columns=['subject', 'reason', 'department', 'source', 'ward_number' ],
                                      model_type="linear",
                                      scenario={'dropOpen':'2021-12-31',
                                                'survivalTimeMin':0})
    test_data_enhanced_lin2 = myBoston311Model_lin2.enhance_data(test_data_2022)
    test_data_cleaned_lin2 = myBoston311Model_lin2.clean_data(test_data_enhanced_lin2)
    linear_test_X_2, linear_test_y_2 = myBoston311Model_lin2.split_data(test_data_cleaned_lin2)


    #check if the function output matches the expected output when reindexed

    test_data = [
        (logistic_test_X_0, tlogistic_test_X_0),
        (logistic_test_X_1, tlogistic_test_X_1),
        (linear_test_X_0, tlinear_test_X_0),
        (linear_test_X_1, tlinear_test_X_1),
        (linear_test_X_2, tlinear_test_X_2),
        (logistic_test_y_0, tlogistic_test_y_0),
        (logistic_test_y_1, tlogistic_test_y_1),
        (linear_test_y_0, tlinear_test_y_0),
        (linear_test_y_1, tlinear_test_y_1),
        (linear_test_y_2, tlinear_test_y_2)
    ]

    for data, expected in test_data:
        if isinstance(data, pd.DataFrame):
            # Sort the DataFrames by index and column names
            data = data.sort_index(axis=0).sort_index(axis=1)
            expected = expected.sort_index(axis=0).sort_index(axis=1)
            # Reset the index to avoid issues with different index types
            data = data.reset_index(drop=True)
            expected = expected.reset_index(drop=True)
            # Compare the DataFrames and assert that they are equal
            #print("Dataframe indices:")
            #print(data.index)
            #print(expected.index)
            #print("Dataframe columns:")
            #print(data.columns)
            #print(expected.columns)
            #diff = data.compare(expected)
            #if not diff.empty:
            #    print(f"DataFrames are different:\n{diff}")
            assert_frame_equal(data, expected, check_dtype=False)
        elif isinstance(data, pd.Series):
            # Sort the Series by index
            #data = data.sort_index(axis=0)
            data = data.rename(None)
            #expected = expected.sort_index(axis=0)
            # Compare the Series and assert that they are equal
            #print("Series indices:")
            #print(data.index)
            #print(expected.index)
            #diff = data.compare(expected)
            #if not diff.empty:
            #    print(f"Series are different:\n{diff}")
            assert_series_equal(data, expected, check_dtype=False)

