from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
import numpy as np
import pickle 
import json
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from kerastuner import HyperParameters
from .Boston311Model import Boston311Model

class Boston311KerasNN(Boston311Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_hyperparameters = None
        self.input_dim = None
        self.batch_size = kwargs.get('batch_size', 32)
        self.patience = kwargs.get('patience', 5)
        self.api_data = kwargs.get('api_data', None)
        self.bin_edges = kwargs.get('bin_edges', None)
        self.bin_labels = kwargs.get('bin_labels', None)
        self.epochs = kwargs.get('epochs', None)


    def save(self, filepath, model_file, properties_file):
        # Save keras model
        self.model.save(filepath + '/' + model_file + '.keras')

        with open(filepath + '/' + properties_file + '.json', 'w') as f:
            save_dict = {
                'feature_columns': self.feature_columns,
                'feature_dict': self.feature_dict,
                'train_date_range': self.train_date_range,
                'predict_date_range': self.predict_date_range,
                'scenario': self.scenario,
                'bin_edges': self.bin_edges,
                'bin_labels': self.bin_labels,
                'epochs': self.epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'input_dim': self.input_dim
            }
            if self.best_hyperparameters is not None :
                save_dict['best_hyperparameters'] = self.best_hyperparameters.get_config()

            json.dump(save_dict, f)


    def load(self, json_file, model_file):
        # Load other properties
        with open(json_file, 'r') as f:
            properties = json.load(f)
            self.feature_columns = properties['feature_columns']
            self.feature_dict = properties['feature_dict']
            self.train_date_range = properties['train_date_range']
            self.predict_date_range = properties['predict_date_range']
            self.scenario = properties['scenario']
            #check if properties has a best_hyperparameters attribute, and if so, load it
            if 'best_hyperparameters' in properties and properties['best_hyperparameters'] is not None:
                self.best_hyperparameters = HyperParameters.from_config(properties['best_hyperparameters'])
    
        self.model = keras.models.load_model(model_file)

    def load_data(self, data=None, train_or_predict='train') :
        return super().load_data(data, train_or_predict)
    
    def enhance_data(self, data, train_or_predict='train'):
        return super().enhance_data(data, train_or_predict)
    
    def apply_scenario(self, data):
        return super().apply_scenario(data)
    
    def clean_data(self, data):
        data = super().clean_data(data)
        for col in data.columns:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype('float64')
        return data
    
    def add_api_data(self, data, api_data):
        data = data.drop_duplicates(subset=['case_enquiry_id'])
        api_data = api_data.drop_duplicates(subset=['case_enquiry_id'])
        data = data.merge(api_data, on='case_enquiry_id', how='inner')
        return data
    
    def clean_data_for_prediction(self, data):
        data = super().clean_data_for_prediction(data)
        for col in data.columns:
            if data[col].dtype == 'bool':
                data[col] = data[col].astype('float64')
        return data
    
    def one_hot_encode_with_feature_dict(self, data):
        return super().one_hot_encode_with_feature_dict(data)
    
    def predict( self, api_data=None, data=None ) :
        if data is None :
            data = self.load_data( train_or_predict='predict' )
        else :
            data = self.load_data( data, train_or_predict='predict' )
        data = self.enhance_data( data, 'predict')
        clean_data = self.clean_data_for_prediction( data )
        if api_data is not None :
            clean_data = self.add_api_data(clean_data, api_data)
            data_limited = data[data['case_enquiry_id'].isin(api_data['case_enquiry_id'])]
        
        X_predict, y_predict = self.split_data( clean_data, self.bin_labels, self.bin_edges )
        y_predict = self.model.predict(X_predict)
        
        return y_predict, data_limited 
    
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
    
    def flatten_and_replace_columns(self, df, column_names):
        new_dfs = []
        
        # Flatten and remove original columns
        for col_name in column_names:
            flattened = np.stack(df[col_name].to_numpy())
            new_df = pd.DataFrame(flattened, columns=[f'{col_name}_{i}' for i in range(flattened.shape[1])])
            new_dfs.append(new_df)
            df.drop([col_name], axis=1, inplace=True)
            
        # Concatenate new columns to the original DataFrame
        df = pd.concat([df] + new_dfs, axis=1)
        return df
    
    def split_data(self, data, bin_labels=None, bin_edges=None) :

        X = data.drop(['survival_time_hours', 'event'], axis=1)
        #if X has a 'case_enquiry_id' column, drop it
        if 'case_enquiry_id' in X.columns :
            X = X.drop(['case_enquiry_id'], axis=1)
        
        if bin_edges is None :
            #keep y as survival_time_hours for regression
            y = data['survival_time_hours']
        else :
            y = pd.cut(data['survival_time_hours'], bins=bin_edges, labels=bin_labels)
        
        return X, y 
        
    def train_model( self, X, y=[], start_nodes=128, end_nodes=64, final_layer_choice=9, final_activation_choice='softmax', epochs=10 ) :
        test_accuracy = 0
        self.model, test_accuracy = self.train_keras_model( X, y, start_nodes, end_nodes, final_layer_choice, final_activation_choice, my_epochs=epochs )
        return test_accuracy

    def train_keras_model ( self, tree_X, tree_y, start_nodes=128, end_nodes=64, final_layer_choice=9, final_activation_choice='softmax', my_epochs=10 ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        self.input_dim = tree_X.shape[1]
        print("input_dim: {}".format(self.input_dim))
        

        # Split into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(tree_X, tree_y, test_size=0.2, random_state=42)

        # Calculate the index for the split
        split_index = int(0.6 * len(tree_X))
        ###
        # Create training and testing sets
        #X_train = tree_X.iloc[:split_index]
        #y_train = tree_y.iloc[:split_index]

        # Make them leaky instead
        #X_train = tree_X
        #y_train = tree_y

        #X_test = tree_X.iloc[split_index:]
        #y_test = tree_y.iloc[split_index:]
        ###

        # Define indices
        indices = np.arange(len(tree_X))

        # For training, take all but every 5th and 4th case
        train_idx = indices[(indices % 5 != 0)]
        X_train = tree_X.iloc[train_idx]
        y_train = tree_y.iloc[train_idx]

        # For validation, take every 4th case
        #val_idx = indices[indices % 5 == 1]
        #X_val = tree_X.iloc[val_idx]
        #y_val = tree_y.iloc[val_idx]

        X_val = tree_X.iloc[split_index:]
        y_val = tree_y.iloc[split_index:]

        # For testing, take every 5th case
        test_idx = indices[indices % 5 == 0]
        X_test = tree_X.iloc[test_idx]
        y_test = tree_y.iloc[test_idx]


        if self.best_hyperparameters is not None:
            model = self.build_model(self.best_hyperparameters)
        else:
            hp = HyperParameters()
            hp.Fixed('start_nodes', value=start_nodes)
            hp.Fixed('end_nodes', value=end_nodes)
            hp.Fixed('final_layer', value=final_layer_choice)
            hp.Fixed('l2_0', value=0.00001)
            hp.Fixed('learning_rate', value=7.5842e-05)
            hp.Fixed('final_activation', value=final_activation_choice)

            # Build the model with the specific hyperparameters
            model = self.build_model(hp)

        print(model.summary())

        #Add Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        # Fit the model
        y_train = pd.get_dummies(y_train)
        y_val = pd.get_dummies(y_val)

        #print debug
        print(type(y_train), y_train.shape)

        y_test = pd.get_dummies(y_test)

        #print debug
        print(type(y_test), y_test.shape)

        print("run fit\n")

        model.fit(X_train, y_train, epochs=my_epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Evaluate the model
        test_loss, test_accuracy, top2_accuracy = model.evaluate(X_test, y_test)
        print('Testing accuracy:', test_accuracy, '\nTop-2 accuracy:', top2_accuracy, '\nTest loss:', test_loss)

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model, test_accuracy
    
    def run_pipeline( self, data_original=None, api_data=None) :
        data = None
        if self.bin_edges is None :
            print("bin_edges is None")  
            self.bin_edges = self.generate_time_bins_fixed_interval(24, 180)
            self.bin_labels = self.generate_bin_labels(self.bin_edges, "over 6 months")
        if self.bin_labels is None :
            print("bin_labels is None")
            last_edge = self.bin_edges[-1]
            overflow_label = self.time_format(last_edge)
            self.bin_labels = self.generate_bin_labels(self.bin_edges, "over " + overflow_label)
        if data_original is None :
            data = self.load_data()
        else :
            data = self.load_data(data_original)
        data = self.enhance_data(data)
        data = self.apply_scenario(data)
        data = self.clean_data(data)
        if api_data is not None :
            data = self.add_api_data(data, api_data)
        #sort before split_data so train_model can take every 5th case for testing
        data = data.sort_values(by='case_enquiry_id')
        X, y = self.split_data(data, self.bin_labels, self.bin_edges)
        if self.epochs is None :
            self.epochs = 2
        test_accuracy = self.train_model( X, y, epochs=self.epochs )
        return test_accuracy
    
    def tune_model( self, X_train, y_train, model_dir, verboseLevel=1):
        print(type(X_train), X_train.shape)
        print(type(y_train), y_train.shape)
        y_train = pd.get_dummies(y_train)
        print(type(y_train), y_train.shape)

        self.input_dim = X_train.shape[1]
        tuner = BayesianOptimization(
            hypermodel=self.build_model,
            objective='val_accuracy',
            max_trials=300,
            num_initial_points=10,
            directory=model_dir,
            project_name='keras_tuning',
            overwrite=True
        )

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        tuner.search(X_train_split, y_train_split,
                    epochs=10,
                    validation_data=(X_val_split, y_val_split),
                    verbose=verboseLevel)
        
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        return best_model, best_hyperparameters

    def build_model( self, hp):
        model = Sequential()

        # Assuming input shape is (steps, input_dim)
        #model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_dim))
        #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        #model.add(Flatten())  # Flatten the sequence to 1D array for the Dense layer

        start_nodes_choice = hp.Choice(f'start_nodes', [128, 256, 512, 1024])
        end_nodes_choice = hp.Choice(f'end_nodes', [16, 32, 64])
        final_layer_choice = hp.Choice(f'final_layer', [16, 32, 64])
        final_activation_choice = hp.Choice(f'final_activation', ['softmax', 'linear'])
        model.add(Dense(start_nodes_choice, input_dim=self.input_dim, activation='relu', kernel_regularizer=l2(hp.Float('l2_0', min_value=1e-5, max_value=1e-1, sampling='LOG'))))
        
        #if hp.Choice(f'batch_normalization', [True, False]):
        #    model.add(BatchNormalization())

        #create a loop to add layers of half the size of the previous layer until the layer size is end_nodes_choice:
        while start_nodes_choice > end_nodes_choice: 
            start_nodes_choice = start_nodes_choice // 2 
            model.add(Dense(start_nodes_choice, activation='relu', kernel_regularizer=l2(hp.Float('l2_0', min_value=1e-5, max_value=1e-1, sampling='LOG'))))
            #add a dropout of .2
            #model.add(Dropout(0.2))
                
        model.add(Dense(final_layer_choice, activation=final_activation_choice))
        
        top2_acc = TopKCategoricalAccuracy(k=2)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG'))
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top2_acc])
        
        return model
