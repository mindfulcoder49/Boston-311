from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
import pickle 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from .Boston311Model import Boston311Model

class Boston311KerasNLP(Boston311Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_hyperparameters = None
        self.input_dim = None


    def save(self, filepath, model_file, properties_file):
        # Save keras model
        self.model.save(filepath + '/' + model_file + '.h5')
        
        # Save other properties
        super().save_properties(filepath, properties_file)

    def load(self, json_file, model_file):

        # Load other properties
        super().load_properties(json_file)
    
        self.model = keras.models.load_model(model_file)

    def load_data(self, train_or_predict='train') :
        return super().load_data(train_or_predict)
    
    def enhance_data(self, data, train_or_predict='train'):
        return super().enhance_data(data, train_or_predict)
    
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
        #if X has a 'case_enquiry_id' column, drop it
        if 'case_enquiry_id' in X.columns :
            X = X.drop(['case_enquiry_id'], axis=1)
        bin_edges = [0, 12, 24, 72, 168, 336, 672, 1344, 2688, 9999999]
        bin_labels = [
            "0-12 hours",      # Less than half a day
            "12-24 hours",     # Half to one day
            "1-3 days",        # One to three days
            "4-7 days",        # Four to seven days
            "1-2 weeks",       # One to two weeks
            "2-4 weeks",       # Two to four weeks
            "1-2 months",      # One to two months
            "2-4 months",      # Two to four months
            "4+ months"        # More than four months
        ]

        y = pd.cut(data['survival_time_hours'], bins=bin_edges, labels=bin_labels)
            
        
        return X, y 
        
    def train_model( self, X, y=[] ) :
        test_accuracy = 0
        self.model, test_accuracy = self.train_keras_model( X, y )
        return test_accuracy

    def train_keras_model ( self, tree_X, tree_y ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(tree_X, tree_y, test_size=0.2, random_state=42)

        if self.best_hyperparameters is not None:
            model = self.build_model(X_train.shape[1], self.best_hyperparameters)
        else:
            model = Sequential()
            model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            
            model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            
            model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            
            model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            
            model.add(Dense(9, activation='softmax'))


        # Initialize the Top-2 accuracy metric
        top2_acc = TopKCategoricalAccuracy(k=2)

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top2_acc])


        #Add Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit the model
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Evaluate the model
        test_loss, test_accuracy, top2_accuracy = model.evaluate(X_test, y_test)
        print('Testing accuracy:', test_accuracy, '\nTop-2 accuracy:', top2_accuracy, '\nTest loss:', test_loss)

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model, test_accuracy
    
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
        test_accuracy = self.train_model( X, y )
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
        start_nodes_choice = hp.Choice(f'start_nodes', [128, 256, 512, 1024])
        end_nodes_choice = hp.Choice(f'end_nodes', [16, 32, 64])
        model.add(Dense(start_nodes_choice, input_dim=self.input_dim, activation='relu', kernel_regularizer=l2(hp.Float('l2_0', min_value=1e-5, max_value=1e-1, sampling='LOG'))))
        #if hp.Choice(f'batch_normalization', [True, False]):
        #    model.add(BatchNormalization())

        #create a loop to add layers of half the size of the previous layer until the layer size is end_nodes_choice:
        while start_nodes_choice > end_nodes_choice: 
            start_nodes_choice = start_nodes_choice // 2 
            model.add(Dense(start_nodes_choice, activation='relu', kernel_regularizer=l2(hp.Float('l2_0', min_value=1e-5, max_value=1e-1, sampling='LOG'))))
        #    if hp.Choice(f'batch_normalization', [True, False]):
        #        model.add(BatchNormalization())
                
        model.add(Dense(9, activation='softmax'))
        
        top2_acc = TopKCategoricalAccuracy(k=2)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG'))
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top2_acc])
        
        return model
