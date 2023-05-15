from boston311 import Boston311Model

my311Model = Boston311Model(train_date_range={'start':'2010-12-31','end':'2024-01-01'},
                            model_type='logistic',
                            feature_columns=['subject', 'reason', 'department', 'source', 'ward_number','type'],
                            scenario={'dropColumnValues': {'source':['City Worker App', 'Employee Generated']},
                                      'dropOpen': '2023-04-13',
                                      'survivalTimeMin':300,
                                      'survivalTimeMax':2678400})
