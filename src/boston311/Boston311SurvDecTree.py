from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pickle 

import Boston311Model

class Boston311SurvDecTree(Boston311Model.Boston311Model):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save(self, filepath, model_file, properties_file):
                
        with open(filepath + '/' + model_file + '.pkl', 'wb') as f:
            pickle.dump(self.model, f)
       
        # Save other properties
        super().save_properties(filepath, properties_file)

    def load(self, json_file, model_file):

        # Load other properties
        super().load_properties(json_file)
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

    def load_data(self, train_or_predict='train') :
        return super().load_data(train_or_predict)
    
    def enhance_data(self, data, train_or_predict='train'):
        data = super().enhance_data(data, train_or_predict)
    
    def apply_scenario(self, data):
        return super().apply_scenario(data)
    
    def clean_data(self, data):
        return super().clean_data(data)
    
    def clean_data_for_prediction(self, data):
        return super().clean_data_for_prediction(data)
    
    def one_hot_encode_with_feature_dict(self, data):
        return super().one_hot_encode_with_feature_dict(data)
    
    def predict( self ) :
        data = self.load_data( 'predict' )
        data = self.enhance_data( data, 'predict')
        clean_data = self.clean_data_for_prediction( data )

        X_predict, y_predict = self.split_data( clean_data )
        y_predict = self.model.predict(X_predict)
        data['survival_prediction'] = y_predict
        return data
    
    def split_data(self, data) :

        X = data.drop(['survival_time_hours', 'event'], axis=1)
        bin_edges = [0, 24, 168, 672, 8736, 1314870]
        bin_labels = ["0-24 hours", "1-7 days","2-4 weeks","1-12 months","over a year"]
        y = pd.cut(data['survival_time_hours'], bins=bin_edges, labels=bin_labels)
            
        
        return X, y 
        
    def train_model( self, X, y=[] ) :
        self.model = self.train_tree_model( X, y )

    def train_tree_model ( self, tree_X, tree_y ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(tree_X, tree_y, test_size=0.2, random_state=42)

        # Initialize the model
        model = DecisionTreeClassifier(random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)

        # Calculate the accuracy on the testing set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print('Testing accuracy:', test_accuracy)

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model
    
    def run_pipeline( self, data_original=None) :
        data = None
        if data_original is None :
            data = self.load_data()
        else :
            data = data_original.copy()
        data = self.enhance_data(data)
        data = self.apply_scenario(data)
        data = self.clean_data(data)
        X, y = self.split_data(data)
        self.train_model( X, y )