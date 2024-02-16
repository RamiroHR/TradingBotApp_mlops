import pandas as pd
import pandas_ta as ta
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import json

class tdbotModel:
    def __init__(self, path='', model_name='model_test'):
        
        self.test_size = 1000
        self.update_file_path(path)
        self.model_name=model_name
    
    def update_file_path(self, file_path):
        self.model_path=os.path.abspath(file_path) # convert the directory into an absolute directory in order to avoid potential bugs during deployments
        # Check if the folder exists
        if not os.path.exists(self.model_path):
            # If it doesn't exist, create it
            os.makedirs(self.model_path)
            print(f"Directory '{self.model_path}' created successfully.")
        else:
            print(f"Directory '{self.model_path}' already exists.")
    
    def get_params(self):
        '''
        Get the parameters for a model on the basis of its name
        '''
        file_name=self.model_name+'_params.json'
        file_path = os.path.join(self.model_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as json_params:
                return json.load(json_params)
        else:
            return False

    def update_params(self, params):
        '''
        Update the parameters to be used for the model
        '''
        file_name=self.model_name+'_params.json'
        file_path = os.path.join(self.model_path, file_name)
        with open(file_path, 'w') as params_file:
            params_file.write(json.dumps(params, indent=4))
            return "Parameters updated"

    def assess_model(self, X, y, params):

        # removal of the NaN lines 
        index_to_drop = X[(X.isna().any(axis=1)) | (y.isna())].index
        X_=X.drop(index_to_drop)
        y_=y.drop(index_to_drop)

        # Initialize KNeighborsRegressor
        #params = self.get_params()
        model = KNeighborsClassifier(**params)

        # Initialize TimeSeriesSplit
        tspl = TimeSeriesSplit(n_splits=10, test_size=self.test_size)

        # Perform cross_validation
        scores = cross_val_score(model, X_.values, y_.values, cv=tspl, scoring='accuracy')

        #print("Cross-validation scores:", scores)
        #print("Mean:", scores.mean())  # Take the mean of the negative scores
        #print("std:", scores.std())  # Take the mean of the negative scores
        return scores

    def train_model(self, X, y):

        # removal of the NaN lines 
        index_to_drop = X[(X.isna().any(axis=1)) | (y.isna())].index
        X=X.drop(index_to_drop)
        y=y.drop(index_to_drop)

        # split train/test
        X_train = X.iloc[:-self.test_size]
        y_train = y.iloc[:-self.test_size]
        X_test = X.iloc[-self.test_size:]
        y_test = y.iloc[-self.test_size:]

        # Initialize KNeighborsRegressor
        params = self.get_params()
        print(params)
        if params: 
            model = KNeighborsClassifier(**params)
            #print("X_train_shape: ", X_train.shape)
            model.fit(X_train, y_train)

            # calculate accuracy
            y_pred_test = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred_test)

            return "Training completed.", model, acc
        else:
            # if there are no parameter, return False
            return "Error. Missing parameters.", None, None
    
    def save_model(self, model):
        try:
            filename = self.model_name+'.joblib'
            file_path=os.path.join(self.model_path,filename)
            joblib.dump(model, file_path)
            return "Model recorded"
        except:
            return "Error with tdbotModel.save_model. Model not recorded."
    
    def load_model(self):
        try:
            filename = self.model_name+'.joblib'
            file_path=os.path.join(self.model_path,filename)
            joblib.dump(model, file_path)
            return "Model recorded"
        except:
            return "Error with tdbotModel.save_model. Model not recorded."

    def get_prediction(self, X):
        filename = self.model_name+'.joblib'
        file_path=os.path.join(self.model_path,filename)
        model = joblib.load(file_path)
        return model.predict(X)[-1]
