from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
from .Boston311Model import Boston311Model

class Boston311LogReg(Boston311Model):

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
        data['event_prediction'] = y_predict
        return data
    
    def split_data(self, data) :

        X = data.drop(['survival_time_hours', 'event'], axis=1) 
        #if X has a 'case_enquiry_id' column, drop it
        if 'case_enquiry_id' in X.columns :
            X = X.drop(['case_enquiry_id'], axis=1)
        y = data['event']
        
        return X, y 
        
    def train_model( self, X, y=[] ) :
        test_acc = 0
        self.model, test_acc = self.train_logistic_model( X, y )
        return test_acc


    def train_logistic_model ( self, logistic_X, logistic_y ) :
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(logistic_X, logistic_y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Build model
        model = keras.Sequential([
            keras.layers.Dense(units=1, input_shape=(X_train.shape[1],), activation='sigmoid')
        ])

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

        # Train model with early stopping
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)

        print('Test accuracy:', test_acc)

        end_time = datetime.now()
        total_time = (end_time - start_time)
        print("Ending Training at {}".format(end_time))
        print("Training took {}".format(total_time))

        return model, test_acc
    
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
        test_acc = self.train_model( X, y )
        return test_acc