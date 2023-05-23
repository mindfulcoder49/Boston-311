from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
from .Boston311Model import Boston311Model

class Boston311LinReg(Boston311Model):

    def save(self, filepath, model_file, properties_file):
        # Save keras model
        self.model.save(filepath + '/' + model_file + '.h5')
        
        # Save other properties
        super().save_properties(filepath, properties_file)

    def load(self, json_file, model_file):

        # Load other properties
        super().load_properties(json_file)
        self.model = keras.models.load_model(model_file)
    
    def predict( self ) :
        data = self.load_data( 'predict' )
        data = self.enhance_data( data, 'predict')
        clean_data = self.clean_data_for_prediction( data )

        X_predict, y_predict = self.split_data( clean_data )
        y_predict = self.model.predict(X_predict)
        data['survival_prediction'] = y_predict
        data['survival_timedelta'] = data['survival_prediction'].apply(lambda x: pd.Timedelta(seconds=(x*3600)))
        data['closed_dt_prediction'] = data['open_dt'] + data['survival_timedelta']
        return data
       
    def split_data(self, data) :
        X = data.drop(['survival_time_hours', 'event'], axis=1) 
        y = data['survival_time_hours']
        
        return X, y 
    
    def train_model( self, X, y=[] ) :
        self.model = self.train_linear_model( X, y )
        
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
