import pandas as pd
import pandas_ta as ta
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import json



#>>>
from pymongo import MongoClient
import bson.binary
from io import BytesIO

# Replace these values with your MongoDB credentials and container details
USERNAME = 'tradingbot_admin'
PASSWORD = 'tradingbot_pass' #os.environ.get('DB_ADMIN_PASS')  #'tradingbot_pass'
CLUSTERNAME = 'tb-cluster-0'

# Connection URI
connection_uri = "mongodb+srv://{username}:{password}@{cluster_name}.ztadynx.mongodb.net/?retryWrites=true&w=majority"
uri = connection_uri.format(
    username=USERNAME,
    password=PASSWORD,
    cluster_name=CLUSTERNAME
)
#<<<

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

##>>>
    # def get_params(self):
    #     '''
    #     Get the parameters for a model on the basis of its name
    #     '''
    #     file_name=self.model_name+'_params.json'
    #     file_path = os.path.join(self.model_path, file_name)
    #     print(file_path)
    #     if os.path.isfile(file_path):
    #         with open(file_path, 'r') as json_params:
    #             print(json_params)
    #             return json.load(json_params)
    #     else:
    #         return False


    def get_params(self):
        '''
        Get the parameters of a model on the basis of its name from mongoDB cluster database
        '''
                
        with MongoClient(uri) as Mclient:
            db = Mclient.models            
            collection = db['model_params']

            model_params = collection.find_one({"model_name": self.model_name})

        if model_params is None:
            # return self.model_name + " not found in Database"
            return False
        else: 
            # Convert ObjectId to string if necessary
            model_params["_id"] = str(model_params["_id"])

        return model_params
#<<<


##>>>
    # def update_params(self, params):
    #     '''
    #     Update the parameters to be used for the model
    #     '''
    #     file_name=self.model_name+'_params.json'
    #     file_path = os.path.join(self.model_path, file_name)
    #     with open(file_path, 'w') as params_file:
    #         params_file.write(json.dumps(params, indent=4))
    #         return "Parameters updated"

    def update_params(self, params):
        '''
        Update the parameters to be used for the model
        '''
        file_name=self.model_name+'_params.json'
        file_path = os.path.join(self.model_path, file_name)
        
        with MongoClient(uri) as Mclient:
            db = Mclient.models            
            collection = db['model_params']

            previous_params = collection.find_one({"model_name": self.model_name})

            # Define the filter to find the document by its "name"
            filter = {"model_name": self.model_name}

            # Define the new values you want to update
            # new_values = {"$set": params["params_model"]}  # Update the values as needed
            new_values = {"$set": params}  # Update the values as needed

            # Update the document
            result = collection.update_one(filter, new_values, upsert=True)
        
            if result.modified_count > 0:
                return "Parameters updated"
            elif result.upserted_id is not None:
                return "Parameters inserted"
            else:
                return "No changes made"
##<<<


    def assess_model_ml_perf(self, X, y, params_model, param_cv):

        # removal of the NaN lines 
        index_to_drop = X[(X.isna().any(axis=1)) | (y.isna())].index
        X_=X.drop(index_to_drop)
        y_=y.drop(index_to_drop)

        # Initialize KNeighborsRegressor
        #params = self.get_params()
        model = KNeighborsClassifier(**params_model)

        # Initialize TimeSeriesSplit
        tspl = TimeSeriesSplit(n_splits=param_cv['n_splits'])

        # Perform cross_validation
        scores = cross_val_score(model, X_.values, y_.values, cv=tspl, scoring='accuracy')

        #print("Cross-validation scores:", scores)
        #print("Mean:", scores.mean())  # Take the mean of the negative scores
        #print("std:", scores.std())  # Take the mean of the negative scores
        return scores


    def assess_model_finance_perf(self, X, y, params_model, param_cv, price_serie):

        # removal of the NaN lines 
        index_to_drop = X[(X.isna().any(axis=1)) | (y.isna())].index
        X_=X.drop(index_to_drop)
        y_=y.drop(index_to_drop)
        price_serie = price_serie.drop(index_to_drop)

        # Initialize TimeSeriesSplit
        tspl = TimeSeriesSplit(n_splits=param_cv['n_splits'])

        # Perform cross_validation
        entry_score_list=np.array([])
        accuracy_score_list=np.array([])

        for train, test in tspl.split(X_):
            #print("%s %s - %s %s" % (len(train), len(test), train[-1], test[-1]))
            # Initialize KNeighborsRegressor
            model = KNeighborsClassifier(**params_model)

            # split train/test data
            X_train = X_.iloc[train]
            y_train = y_.iloc[train]
            X_test = X_.iloc[test]
            y_test = y_.iloc[test]
            price_test = price_serie.iloc[test]

            # train
            model.fit(X_train.values, y_train)

            # pred
            y_test_pred = model.predict(X_test.values)

            # calculate financial perf
            ratio = self.get_entry_score(y_test_pred, price_test)
            entry_score_list = np.append(entry_score_list, ratio)

            # accuracy
            acc = accuracy_score(y_test.values, y_test_pred)
            accuracy_score_list = np.append(accuracy_score_list, acc)
        
        # clean list by removing the NaN values (to avoid errors)
        arg_nan = np.argwhere(np.isnan(entry_score_list))
        entry_score_list = np.delete(entry_score_list, arg_nan)
        accuracy_score_list = np.delete(accuracy_score_list, arg_nan)

        # create output data
        eps_score = {
            'cv-list': list(entry_score_list),
            'mean': np.nanmean(entry_score_list),
            'std': np.nanstd(entry_score_list),
        }

        acc_score = {
            'cv-list': list(accuracy_score_list),
            'mean': np.nanmean(accuracy_score_list),
            'std': np.nanstd(accuracy_score_list),
        }

        scores={
            'entry_score': eps_score,
            'accuracy_score': acc_score
        }
        
        return scores

    def train_model(self, X, y, price_serie):

        # removal of the NaN lines 
        index_to_drop = X[(X.isna().any(axis=1)) | (y.isna())].index
        X=X.drop(index_to_drop)
        y=y.drop(index_to_drop)

        # get the parameters for this model_name
        params = self.get_params()

        # split train/test
        n_split = params['params_cv']['n_splits']
        self.test_size = len(X)//(n_split+1)
        X_train = X.iloc[:-self.test_size]
        y_train = y.iloc[:-self.test_size]
        X_test = X.iloc[-self.test_size:]
        y_test = y.iloc[-self.test_size:]
        price_test = price_serie.loc[y_test.index]

        # Initialize KNeighborsRegressor
        
        if params: 
            model = KNeighborsClassifier(**params['params_model'])
            #print("X_train_shape: ", X_train.shape)
            model.fit(X_train, y_train)

            # calculate accuracy
            y_test_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)
            entry_score = self.get_entry_score(y_test_pred, price_test)

            return "Training completed.", model, acc, entry_score
        else:
            # if there are no parameter, return False
            return "Error. Missing parameters.", None, None, None

##>>>    
    # def save_model(self, model):
    #     try:
    #         filename = self.model_name+'.joblib'
    #         file_path=os.path.join(self.model_path,filename)
    #         joblib.dump(model, file_path)
    #         return "Model recorded"
    #     except:
    #         return "Error with tdbotModel.save_model. Model not recorded."
    def save_model(self, model):
        try:
            filename = self.model_name+'.joblib'
            file_path=os.path.join(self.model_path,filename)
            #joblib.dump(model, file_path)
        
            with MongoClient(uri) as Mclient:
                db = Mclient.models            
                collection = db['trained_models']
                
                # Serialize the model to a binary format
                with open(file_path, 'wb') as f:
                    joblib.dump(model, f)

                # Read the serialized model bytes
                with open(file_path, 'rb') as f:
                    model_bytes = bson.binary.Binary(f.read())
                
                # Data to insert
                doc = {"model_name": self.model_name, "model": model_bytes}

                # Insert the model bytes into the collection
                collection.insert_one(doc)
            return "Model recorded"
        
        except:
            return "Error with tdbotModel.save_model. Model not recorded."
##<<<

##>>>
    # def load_model(self):
    #     try:
    #         filename = self.model_name+'.joblib'
    #         file_path=os.path.join(self.model_path,filename)
    #         joblib.dump(model, file_path)
    #         return "Model recorded"
    #     except:
    #         return "Error with tdbotModel.save_model. Model not recorded."
    def load_model(self):
        try:
            # filename = self.model_name+'.joblib'
            # file_path=os.path.join(self.model_path,filename)
            # joblib.dump(model, file_path)
            with MongoClient(uri) as Mclient:
                db = Mclient.models            
                collection = db['trained_models']
                doc = collection.find_one({"model_name": self.model_name})

                model_bytes = doc["model"]
                
                # Create a BytesIO object to store the binary data, then load to correctly handle utf-encoding
                model_buffer = BytesIO(model_bytes)
                model = joblib.load(model_buffer)        
            return {"Model_recovered": model}

        except:
            return "Error with tdbotModel.load_model. Model not recovered."
##<<<        


##>>>
    # def get_prediction(self, X):
    #     filename = self.model_name+'.joblib'
    #     file_path=os.path.join(self.model_path,filename)
    #     model = joblib.load(file_path)
    #     return model.predict(X)[-1]
    def get_prediction(self, X):
        # filename = self.model_name+'.joblib'
        # file_path=os.path.join(self.model_path,filename)
        model = self.load_model()["Model_recovered"]
        return model.predict(X)[-1]
##<<<

    # create a function to assess the financial performance
    def get_entry_score(self, y_pred, price_serie):
        '''
        This function calculation the average entry price in the market with a standard daily dca vs a machine learning dca.
        The purpose is to assess the performance of the model.

        y_pred: pandas Series with datime index and binary prediction
        price_serie: pandas Series with datetime index and price action
        '''
        dca_daily_entry_price = price_serie.mean() # average entry price for a daily buy order

        index_1 = np.where(y_pred==1)[0]
        
        if len(index_1)>0:
            dca_tdbot_entry_price = price_serie.iloc[index_1].mean() # average entry price for a buy order on the basis of the trading bot signal
            entry_ratio = dca_tdbot_entry_price / dca_daily_entry_price - 1 # If <0, this ratio means that the entry price was better than doing a random daily investment
        else:
            dca_tdbot_entry_price = np.nan
            entry_ratio = np.nan
        
        return entry_ratio