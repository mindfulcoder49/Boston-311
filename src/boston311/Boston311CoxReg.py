from sklearn.model_selection import train_test_split
from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pickle 
from .Boston311Model import Boston311Model

class Boston311CoxReg(Boston311Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        risks = self.model.predict_partial_hazard(clean_data) 
        survival_function = self.model.predict_survival_function(clean_data)
        median_survival_times = self.model.predict_median(clean_data)
        return risks, survival_function, median_survival_times
    
    def split_data(self, data) :
        return data
    
    def train_model( self, X, y=[] ) :
        self.model = self.train_cox_model(X)
        
    def train_cox_model(self, data):
        start_time = datetime.now()
        print("Starting Training at {}".format(start_time))

        # Split the data into a training set, a validation set, and a test set
        #df_temp, test_df = train_test_split(data, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

        # Fit the Cox proportional hazards model
        model = CoxPHFitter()
        model.fit(train_df, duration_col='survival_time_hours', event_col='event')

        # Predict the risk on the validation set and evaluate
        val_duration = val_df.pop('survival_time_hours')
        val_event_observed = val_df.pop('event')
        val_predictions = model.predict_partial_hazard(val_df)
        c_index = concordance_index(val_duration, -val_predictions, val_event_observed)
        print(f"Concordance Index on Validation Set: {c_index}")

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
        self.train_model( data )