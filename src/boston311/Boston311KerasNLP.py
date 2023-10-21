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
from kerastuner import HyperParameters
from .Boston311Model import Boston311Model

class Boston311KerasNLP(Boston311Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_hyperparameters = None
        self.input_dim = None
        self.batch_size = 32


    def save(self, filepath, model_file, properties_file):
        # Save keras model
        self.model.save(filepath + '/' + model_file + '.h5')
        
        # Save other properties
        super().save_properties(filepath, properties_file)

    def load(self, json_file, model_file):

        # Load other properties
        super().load_properties(json_file)
    
        self.model = keras.models.load_model(model_file)

    def load_data(self, data=None, train_or_predict='train') :
        return super().load_data(data, train_or_predict)
    
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
        
    def train_model( self, X, y=[], start_nodes=128, end_nodes=64, final_layer_choice=9, final_activation_choice='softmax' ) :
        test_accuracy = 0
        self.model, test_accuracy = self.train_keras_model( X, y, start_nodes, end_nodes, final_layer_choice, final_activation_choice )
        return test_accuracy

    def train_keras_model ( self, tree_X, tree_y, start_nodes=128, end_nodes=64, final_layer_choice=9, final_activation_choice='softmax', my_epochs=10 ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        self.input_dim = tree_X.shape[1]

        # Split into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(tree_X, tree_y, test_size=0.2, random_state=42)

        # Calculate the index for the split
        split_index = int(0.8 * len(tree_X))

        # Create training and testing sets
        X_train = tree_X.iloc[:split_index]
        y_train = tree_y.iloc[:split_index]

        X_test = tree_X.iloc[split_index:]
        y_test = tree_y.iloc[split_index:]

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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit the model
        y_train = pd.get_dummies(y_train)

        #print debug
        print(type(y_train), y_train.shape)

        y_test = pd.get_dummies(y_test)

        #print debug
        print(type(y_test), y_test.shape)

        print("run fit\n")

        model.fit(X_train, y_train, epochs=my_epochs, batch_size=self.batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

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
        final_layer_choice = hp.Choice(f'final_layer', [16, 32, 64])
        final_activation_choice = hp.Choice(f'final_activation', ['softmax', 'linear'])
        model.add(Dense(start_nodes_choice, input_dim=self.input_dim, activation='relu', kernel_regularizer=l2(hp.Float('l2_0', min_value=1e-5, max_value=1e-1, sampling='LOG'))))
        #if hp.Choice(f'batch_normalization', [True, False]):
        #    model.add(BatchNormalization())

        #create a loop to add layers of half the size of the previous layer until the layer size is end_nodes_choice:
        while start_nodes_choice > end_nodes_choice: 
            start_nodes_choice = start_nodes_choice // 2 
            model.add(Dense(start_nodes_choice, activation='relu', kernel_regularizer=l2(hp.Float('l2_0', min_value=1e-5, max_value=1e-1, sampling='LOG'))))
        #    if hp.Choice(f'batch_normalization', [True, False]):
        #        model.add(BatchNormalization())
                
        model.add(Dense(final_layer_choice, activation=final_activation_choice))
        
        top2_acc = TopKCategoricalAccuracy(k=2)
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG'))
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top2_acc])
        
        return model
