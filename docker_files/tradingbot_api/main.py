from fastapi import FastAPI, Depends, HTTPException, status #, Query, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional, List
from pydantic import BaseModel, Field #, validator, ValidationError
from enum import Enum
from src.data.make_dataset import getData
from datetime import datetime, timezone
import time
import json
import numpy as np
from pymongo import MongoClient
import os
import requests
import pandas as pd

from src.models.train_model import tdbotModel
from src.data.read_dataset import openFile
from src.features.build_features import getFeatures, getTarget


## instantiate the api and define the tags meatadata & security protocol
tags_metadata = [
    {'name': 'Public', 'description':'Methods available to everyone'},
    {'name': 'Users', 'description':'Methods available for registered user'},
    {'name': 'Admins', 'description':'Methods available for administrators'}
]


api = FastAPI(
    title = "Trading Bot App",
    description = "Enhance profits and manage risks with AI assisted investment strategies.",
    version = '1.0.0',
    openapi_tags = tags_metadata
)

raw_data_path = "data/raw"
models_path = "models"


#################################################################################
#### Login protocols 
#################################################################################


### Set up connection to users/admins database

# from src.data.make_all_users_database import create_all_users_database
# create_all_users_database()

# users_db = dbm.get_registered_users()
# admins_db = dbm.get_admin_users()


# ### Define verification functions:
# security = HTTPBasic()

# def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
#     username = credentials.username
    
#     if not(users_db.get(username)) or not(credentials.password == users_db[username]['password']):        raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect user credentials",
#             headers={"WWW-Authenticate": "Basic"},
#         )
#     return username


# def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
#     username = credentials.username

#     if not(admins_db.get(username)) or not(credentials.password == admins_db[username]['password']):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect admin credentials",
#             headers={"WWW-Authenticate": "Basic"},
#         )
#     return username


######################################################################################
########################### test microservice architecture ###########################
######################### -- detach verification process -- ##########################
######################################################################################


### Define verification functions:
security = HTTPBasic()

def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username

    # Make a request to the userdatabase API to verify the user credentials
    # url = 'http://localhost:8001/verify-user'  # as defined by the deployment& & service
    url = 'http://service-tb:8001/verify-user'  # as defined by the deployment& & service
    
    response = requests.get(url, auth=(username, credentials.password))
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username

    # Make a request to the userdatabase API to verify the user credentials
    # url = 'http://localhost:8001/verify-admin'  # as defined by the deployment& & service
    url = 'http://service-tb:8001/verify-admin'  # as defined by the deployment& & service
    
    response = requests.get(url, auth=(username, credentials.password))
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username


######################################################################################
############################# test connection to MongoDB #############################
######################### -- Add only some basic endpoints -- ########################
######################################################################################



# Replace these values with your MongoDB credentials and container details
USERNAME = 'tradingbot_admin'
PASSWORD = os.environ.get('DB_ADMIN_PASS')
CLUSTERNAME = 'tb-cluster-0'

# Connection URI
connection_uri = "mongodb+srv://{username}:{password}@{cluster_name}.ztadynx.mongodb.net/?retryWrites=true&w=majority"
uri = connection_uri.format(
    username=USERNAME,
    password=PASSWORD,
    cluster_name=CLUSTERNAME
)


#################################################################################
########################## Define Public Methods ################################
#################################################################################


###========================= app status =========================###
@api.get('/app_status', name = 'Quick test if the API is running',
                    tags = ['Public'])
async def get_app_status():
    return {'status' : 'App is available'}


###==================== Test database connection =======================###
@api.get('/check-db-connection', name = 'Check database connection',
                    tags = ['Public'])
async def check_MongoDB_connection():
    """
    Test the correct connection to the MongoDB database
    """
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
        return {"status": "Pinged your deployment. You successfully connected to MongoDB Atlas!"}
    except Exception as e:
        # If there's an exception, display the error
        return {"error": str(e)}


###========================== sign up ==========================###
class User(BaseModel):
    username: str
    password: str
 
@api.post('/sign_up', name = 'Create new user',
                    description = 'Add new user to the users database.',
                    tags = ['Public'])
async def post_new_user(user: User):
    new_user = {
            "username": user.username,
            "password": user.password
            }
        
    with MongoClient(uri) as client:
        # Access the database and collection
        db = client.api_login_credentials
        collection = db.users
        
        # Check if the username already exists
        if collection.find_one({"username": user.username}):
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # insert data in MongoDB 
        result = collection.insert_one(new_user)
        
        # Check if the insertion was successful
        if result.acknowledged:
            return {"message": "User added successfully", "new_user": user.username}
        else:
            raise HTTPException(status_code=500, detail="Failed to add user")
     

##================================== check actual price hist =====================================###
# Predefined 'assets' present in the database.
allowed_assets = ['BTCUSDT', 'ETHUSDT']
Asset = Enum('Assets', {item: item for item in allowed_assets}, type = str)

# Predefined 'intervals' present in the database.
allowed_interval = ['1h', '4h', '1d']
Intervals = Enum('Assets', {item: item for item in allowed_interval}, type = str)


@api.get('/check_actual_price_hist', name = 'check if price exists for an asset',
                    description = 'Check if price exists in our database for a specific asset',
                    tags = ['Public'])
async def check_actual_price_hist(
    asset: Asset,
    interval: Intervals):

    open_csv = openFile(raw_data_path, asset, interval)
    if open_csv.is_data() == False:
        raise HTTPException(status_code=404, detail="No data available")
    else:
        data = open_csv.data

        # get the first date of the dataset
        first_date = data.iloc[0].openT
        first_date = datetime.fromtimestamp(first_date/1000)
        first_date = first_date.strftime("%d/%m/%Y %H:%M:%S")

        # get the last date of the dataset
        last_date = data.iloc[-1].closeT
        last_date = datetime.fromtimestamp(last_date/1000) 
        last_date = last_date.strftime("%d/%m/%Y %H:%M:%S")

        output = {
                    'asset': asset, 
                    'interval': interval, 
                    'first_open_date': first_date, 
                    'last_close_date': last_date
                }
    
    return output


##========================== update_price_hist ==========================###


@api.put('/update_price_hist', name = 'Fetch Asset Price History',
                    description = 'Fetch asset specific data over a selected period of time and record it.',
                    tags = ['Public'])
async def update_price_hist(
    asset: Asset,
    interval: Intervals
    ):

    # set the parameters
    params = {
        'pair': asset,
        'interval': interval,
        'pitch': 1000,
        'start_date': int(datetime(year=2017, month=7, day=1, hour=0, minute=0, tzinfo=timezone.utc).timestamp()*1000), # default: start date in july 2017 (Binance launch)
        'end_date': int(time.time()*1000) # default: actual time to get the up to date data
    }

    get_data = getData(**params)
#>>>    get_data.update_file_path(raw_data_path)
    data = get_data.getKlines()

    # get the first date of the dataset
    first_date = data.iloc[0].openT
    first_date = datetime.fromtimestamp(first_date/1000)
    first_date = first_date.strftime("%d/%m/%Y %H:%M:%S")

    # get the last date of the dataset
    last_date = data.iloc[-1].closeT
    last_date = datetime.fromtimestamp(last_date/1000) 
    last_date = last_date.strftime("%d/%m/%Y %H:%M:%S")

    return {'asset': asset, 
            'first_open_date': first_date, 
            'last_close_date': last_date}
    




##========================== get_price_hist ==========================###
@api.get('/get_price_hist', name = 'Get the price history between 2 dates',
                    description = 'Get the price history between 2 dates for a specific asset',
                    tags = ['Public'])
async def get_price_hist(
    asset: Asset,
    interval: Intervals,
    date_start: datetime,
    date_end: datetime):
    """
    Return historical data for a given asset
    """
    try:
        client = MongoClient(uri)

        db = client['asset_price_hist']

        # get asset data
        db_name = f"{asset}-{interval}-raw"
        collection = db[db_name]

        # Convert start and end dates to milliseconds since epoch
        start_timestamp = int(date_start.timestamp() * 1000)
        end_timestamp = int(date_end.timestamp() * 1000)

        # Query MongoDB to filter data between start_date and end_date
        query = {"openT": {"$gte": start_timestamp, "$lte": end_timestamp}}
        cursor = collection.find(query)

        df = pd.DataFrame(list(cursor))

        if len(df)>0:
            df.drop(columns=['_id'], inplace=True)

        output=df.to_json(orient="records")
        output = json.loads(output)

        return output
    except Exception as e:
        # If there's an exception, display the error
        return {"error": str(e)}




##=============================== get targets (for plotting) ===============================###
@api.post('/get_target', name = 'Get the target',
                    description = 'Get the target on the basis of the input parameters and a list of prices.',
                    tags = ['Public'])
async def get_target(
    price_list: List[float],
    target_ema_length: int = 7,
    target_diff_length: int = 7,
    target_pct_threshold: float = 0.2 ):
    
    y_transform = getTarget(target_ema_length, target_diff_length, target_pct_threshold, True)
    y, threshold = y_transform.fit_transform(pd.Series(price_list))
    
    return y.tolist()

def get_existent_models():
    names = []
    with MongoClient(uri) as client:
        db = client.models
        collection = db.trained_models
        ## query:
        names = [item["model_name"] for item in collection.find()]  # Assuming model_name is a field in your collection
    return names


##================================== check if model exists in DB ==================================###
@api.get('/check_model_exists', name = 'Check if a model exists and has been trained',
                    description = 'Check if model exists in our database for a specific asset and interval',
                    tags = ['Public'])
async def check_model_exists(
    asset: Asset,
    interval: Intervals,
    model_name: str = 'model_test'):

    tm = tdbotModel(models_path, model_name, asset, interval)
    if tm.load_model()['model_exist'] == False:
        names = get_existent_models()
        raise HTTPException(status_code=404, detail={"False":"This model is not available.",
                                                     "Available models": names}
                            )

    return True


#################################################################################
######################## Define Registered Users Methods ########################
#################################################################################


##========================== verify user credentials ==========================###
@api.get('/verify_user', name = 'User login verification',
                         description = 'Verify if user is registered in the users database',
                         tags = ['Users'])
async def verify_user_route(username: str = Depends(verify_user)):
    """
    Returns confirmation is login protocol is succesfull
    """
    
    response = '{} is a registered user'.format(username)
    
    return response


#========================== Get model prediction (singular) ==========================###
@api.get('/prediction', name = 'Get final prediction',
                         description = 'Get the prediction with the actual data. Return "BUY" or "WAIT".',
                         tags = ['Users'])
async def get_prediction(
                    asset: Asset,
                    interval: Intervals,
                    model_name: str = 'model_test',
                    username: str = Depends(verify_user)
                    ):
    """
    Returns confirmation if login protocol is succesfull
    """
    
    # read the dataset
    open_csv = openFile(raw_data_path, asset, interval)
    if open_csv.is_data() == False:
        raise HTTPException(status_code=404, detail="No data available. Fetch the data first.")
    df = open_csv.data

    # open a instance of tdbotModel
    tm = tdbotModel(models_path, model_name, asset, interval)

    # read the params
    params = tm.get_params()
    if params==False:
        raise HTTPException(status_code=404, detail="No parameters available. Use /update_model_params to create the parameters.")

    # once we have the data AND the parameters, we can perform the training

    # preprocessing
    features_params=params['params_features_eng']

    # get the features
    X_transform = getFeatures(features_params['features_length'], features_params['features_factor'])
    X = X_transform.fit_transform(df.close)

    print(X.iloc[-1:].shape)
    pred = tm.get_prediction(X.iloc[-1:])
    if pred['pred_exist']==False:
        raise HTTPException(status_code=404, detail="No model trained. Use /train_model to train the model.")
    
    response = "BUY" if pred['prediction']==1 else "WAIT"
    
    return response


##========================== Get predictionS (plural!) ==========================###
@api.get('/predictions', name = 'Get predictions (during training)',
                         description = 'Get the prediction with the actual data. Return "BUY" or "WAIT".',
                         tags = ['Users'])
async def get_predictions(
                    asset: Asset,
                    interval: Intervals,
                    model_name: str = 'model_test',
                    date_start: Optional[datetime] = None,
                    date_end: Optional[datetime] = None,
                    username: str = Depends(verify_user)):
    """
    Returns confirmation if login protocol is succesfull
    """
    
    # read the dataset
    open_csv = openFile(raw_data_path, asset, interval)
    if open_csv.is_data() == False:
        raise HTTPException(status_code=404, detail="No data available. Fetch the data first.")
    df = open_csv.data

    if date_start != None and date_end != None:
        # if we want the prediction for a time range, we can define a filter
        # Convert start and end dates to milliseconds since epoch
        start_timestamp = int(date_start.timestamp() * 1000)
        end_timestamp = int(date_end.timestamp() * 1000)
        index_filter = df[(df.openT>=start_timestamp) & (df.openT<=end_timestamp)].index
    else:
        index_filter = np.array([df.index[-1]])

    # open a instance of tdbotModel
    tm = tdbotModel(models_path, model_name, asset, interval)

    # read the params
    params = tm.get_params()
    if params==False:
        raise HTTPException(status_code=404, detail="No parameters available. Use /update_model_params to create the parameters.")

    # once we have the data AND the parameters, we can perform the training

    # preprocessing
    features_params=params['params_features_eng']

    # get the features
    X_transform = getFeatures(features_params['features_length'], features_params['features_factor'])
    X = X_transform.fit_transform(df.close)

    pred = tm.get_predictions(X.loc[index_filter])
    if pred['pred_exist']==False:
        raise HTTPException(status_code=404, detail="No model trained. Use /train_model to train the model.")
    
    # Let's prepare the response dict
    response = {}
    response['pred_exist'] = True
    response['datetime_index'] = index_filter.tolist() # list of index
    response['pred'] = pred['prediction'].tolist() # list of prediction: 0 or 1
    
    return response

#################################################################################
######################## Define Registered Admin Methods ########################
#################################################################################


class Params(BaseModel):
    features_length: int = Field(default=7)
    features_factor: int = Field(default=10)
    target_ema_length: int = Field(default=7)
    target_diff_length: int = Field(default=14)
    target_pct_threshold: float = Field(default=0.2)
    tdbmodel_n_neighbors: int = Field(default=30)
    tdbmodel_weights: str = Field(default="uniform")
    cv_n_splits: int = Field(default=5)

def get_params_features_eng(params):
    # function to update the input data if incorrect
    params_checked = {}
    params_checked["features_length"] = params.features_length
    params_checked["features_factor"] = params.features_factor
    params_checked["target_ema_length"] = params.target_ema_length
    params_checked["target_diff_length"] = params.target_diff_length
    params_checked["target_pct_threshold"] = params.target_pct_threshold if (type(params.target_pct_threshold)==float) and (params.target_pct_threshold>0) and (params.target_pct_threshold<1) else 0.1
    return params_checked

def get_params_model(params):
    # function to update the input data if incorrect
    params_checked = {}
    params_checked["n_neighbors"] = params.tdbmodel_n_neighbors if (type(params.tdbmodel_n_neighbors)==int and params.tdbmodel_n_neighbors<36) else 30
    params_checked["weights"] = params.tdbmodel_weights if params.tdbmodel_weights in ["uniform", "distance"] else "uniform"
    return params_checked

def get_params_cv(params):
    # function to update the input data if incorrect
    params_checked = {}
    params_checked["n_splits"] = params.cv_n_splits if type(params.cv_n_splits)==int else 5
    return params_checked


##======================== Get the model parameters ========================###
@api.get('/get_model_params', name = "Get the parameters of a model",
                         description = 'Verify the parameters actually in use for a given model (using the model_name as an ID)',
                         tags = ['Admins'])
async def get_model_params(
                            asset: Asset,
                            interval: Intervals,
                            model_name: str = 'model_test',
                            username: str = Depends(verify_admin)
                            ):
    """
    Returns the parameters of the model
    """

    tm = tdbotModel(models_path, model_name, asset, interval)
    response = tm.get_params()

    if response is False:
        # return 404 code if no parameters
        raise HTTPException(status_code=404, detail="No parameters available")
    
    return response


##======================== Update the model parameters ========================###
@api.put('/update_model_params', name = "Create or Update the parameters of a model",
                         description = 'Create or Update the parameters for a model',
                         tags = ['Admins'])
async def update_model_params(
                            asset: Asset,
                            interval: Intervals,
                            params: Params,
                            username: str = Depends(verify_admin),
                            model_name: str = 'model_test'
                            ):
    """
    Update the parameters of the model and return the actual parameters
    """

    # post-process params
    pp_params = {
        "params_features_eng": get_params_features_eng(params),
        "params_model": get_params_model(params),
        "params_cv": get_params_cv(params)
    }
    print(pp_params)
    tm = tdbotModel(models_path, model_name, asset, interval)
    
    response = {}
    response['model_name'] = model_name
    response['previous_parameters'] = tm.get_params()
    response['response'] = tm.update_params(pp_params)
    response['new_parameters'] = tm.get_params()

    return response


##============================= assess a model  =============================###
@api.post('/assess_ml_performance', name = 'Assess the accuracy of a model',
                         description = "Assess the accuracy of a parameters' set with a cross-validation. This endpoint does not record any file.",
                         tags = ['Admins'])
async def assess_ml_performance(
                    asset: Asset,
                    interval: Intervals,
                    params: Params,
                    username: str = Depends(verify_admin)
                    ):
    """
    Returns the parameters of the model
    """

    # read the dataset
    open_csv = openFile(raw_data_path, asset, interval)
    df = open_csv.data

    # preprocessing
    features_params=get_params_features_eng(params)

    # get the features
    X_transform = getFeatures(features_params['features_length'], features_params['features_factor'])
    X = X_transform.fit_transform(df.close)

    # get the target (float)
    y_transform = getTarget(features_params['target_ema_length'], features_params['target_diff_length'], features_params['target_pct_threshold'], True)
    y, threshold = y_transform.fit_transform(df.close)

    # dataset info
    input_info = {}
    input_info['dataset_len'] = X.shape[0]
    input_info['nb_features'] = X.shape[1]
    input_info['target_0'] = len(y[y==0])
    input_info['target_1'] = len(y[y==1])
    input_info['target_1_threshold'] = threshold

    # open a instance of tdbotModel
    tm = tdbotModel(models_path, 'assess', asset, interval)
    
    # get all the parameters in the relevant format
    params_model=get_params_model(params)
    params_cv=get_params_cv(params)

    # assess model
    accs = tm.assess_model_ml_perf(X, y, params_model, params_cv)

    response={}
    response['dataset_info'] = input_info
    response['features_parameters'] = features_params
    response['model_parameters'] = params_model
    response['cv_parameters'] = params_cv
    response['accuracy_list'] = list(accs)
    response['accuracy_mean'] = accs.mean()
    response['accuracy_std'] = accs.std()
    
    return response



##==================== assess a model / with financial indicator ==================###
@api.post('/assess_financial_performance', name = 'Assess the financial performance of a model',
                         description = "Assess the financial performance of a parameters' set with a cross-validation. This endpoint does not record any file.",
                         tags = ['Admins'])
async def assess_financial_performance(
                    asset: Asset,
                    interval: Intervals,
                    params: Params,
                    username: str = Depends(verify_admin)
                    ):
    """
    Returns the results of the assessment
    """

    # read the dataset
    open_csv = openFile(raw_data_path, asset, interval)
    df = open_csv.data

    # preprocessing
    features_params=get_params_features_eng(params)

    # get the features
    X_transform = getFeatures(features_params['features_length'], features_params['features_factor'])
    X = X_transform.fit_transform(df.close)

    # get the target (float)
    y_transform = getTarget(features_params['target_ema_length'], features_params['target_diff_length'], features_params['target_pct_threshold'], True)
    y, threshold = y_transform.fit_transform(df.close)

    # dataset info
    input_info = {}
    input_info['dataset_len'] = X.shape[0]
    input_info['nb_features'] = X.shape[1]
    input_info['target_0'] = len(y[y==0])
    input_info['target_1'] = len(y[y==1])
    input_info['target_1_threshold'] = threshold

    # open a instance of tdbotModel
    tm = tdbotModel(models_path, 'assess', asset, interval)
    
    # get all the parameters in the relevant format
    params_model=get_params_model(params)
    params_cv=get_params_cv(params)

    # assess model
    scores = tm.assess_model_finance_perf(X, y, params_model, params_cv, df.close)

    response={}
    response['dataset_info'] = input_info
    response['features_parameters'] = features_params
    response['model_parameters'] = params_model
    response['cv_parameters'] = params_cv
    response['scores'] = scores
    
    return response



##=========================== train the model =========================###

@api.put('/train_model', name = 'Train the model',
                         description = 'Train the model and save it.',
                         tags = ['Admins'])
async def train_model(
                    asset: Asset,
                    interval: Intervals,
                    model_name: str = 'model_test',
                    username: str = Depends(verify_admin)
                    ):
    """
    Returns the parameters of the model
    """

    # read the dataset
    open_csv = openFile(raw_data_path, asset, interval)
    if open_csv.is_data() == False:
        raise HTTPException(status_code=404, detail="No data available. Fetch the data first.")
    df = open_csv.data

    # open a instance of tdbotModel
    tm = tdbotModel(models_path, model_name, asset, interval)

    # read the params
    params = tm.get_params()
    if params==False:
        raise HTTPException(status_code=404, detail="No parameters available. Use /update_model_params to create the parameters.")

    # once we have the data AND the parameters, we can perform the training

    # preprocessing
    features_params=params['params_features_eng']

    # get the features
    X_transform = getFeatures(features_params['features_length'], features_params['features_factor'])
    X = X_transform.fit_transform(df.close)

    # get the target (float)
    y_transform = getTarget(features_params['target_ema_length'], features_params['target_diff_length'], features_params['target_pct_threshold'], True)
    y, threshold = y_transform.fit_transform(df.close)

    # train the model
    training_response, model, acc, entry_score = tm.train_model(X, y, df.close)
    
    # create the response to return
    response={}
    response['model_name'] = model_name
    response['training_response'] = training_response
    response['recording_response'] = tm.save_model(model)
    response['entry_score'] = entry_score
    response['accuracy'] = acc
    
    return response