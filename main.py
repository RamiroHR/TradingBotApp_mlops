from fastapi import FastAPI, Depends, HTTPException, status, Query, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum
from src.data.make_dataset import getData
from datetime import datetime, timezone
import time
import json
import database_management_tools as dbm
import numpy as np

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

from src.data.make_all_users_database import create_all_users_database
create_all_users_database()

users_db = dbm.get_registered_users()
admins_db = dbm.get_admin_users()


### Define verification functions:
security = HTTPBasic()

def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    
    if not(users_db.get(username)) or not(credentials.password == users_db[username]['password']):        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username

    if not(admins_db.get(username)) or not(credentials.password == admins_db[username]['password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username



#################################################################################
#### [ ] Define Public Methods
#################################################################################

#### [ ] /app_status
@api.get('/app_status', name = 'Quick test if the API is running',
                    tags = ['Public'])
async def get_app_status():
    return {'status' : 'App is available'}


#### [ ] /sign_up
class User(BaseModel):
    username: str
    password: str
 
@api.post('/sign_up', name = 'Create new user',
                    description = 'Add new user to the users database.',
                    tags = ['Public'])
async def post_new_user(user: User):
    
    new_user = {
        user.username: {
            "username": user.username,
            "password": user.password
            }
    }
    
    dbm.write_database(new_user)
    
    return user
     


#### [ ] /price_hist

# Predefined 'assets' present in the database.
allowed_assets = ['BTCUSDT', 'ETHUSDT']
Asset = Enum('Assets', {item: item for item in allowed_assets}, type = str)

# Predefined 'intervals' present in the database.
allowed_interval = ['1h', '4h', '1d']
Intervals = Enum('Assets', {item: item for item in allowed_interval}, type = str)

@api.get('/price_hist', name = 'Fetch Asset Price History',
                    description = 'Fetch asset specific data over a selected period of time and record it.',
                    tags = ['Public'])
async def get_price_hist(
    asset: Asset,
    interval: Intervals
    # start_date: str = Query(None, description="Select earliest date JJ/MM/YYY. Default : 01/07/2017 (Binance launch)."),
    # end_date: str = Query(None, description="Select end date JJ/MM/YYY. Must be after the start date")
    ):
    
    """
    [ ] To complete code. Placeholder.
    [ ] verify end date > start date.
    [ ] Import statique data from asset database (predefined with the binance API)
    """

    # set the parameters
    params = {
        'pair': asset,
        'interval': interval,
        'pitch': 1000,
        'start_date': int(datetime(year=2017, month=7, day=1, hour=0, minute=0, tzinfo=timezone.utc).timestamp()*1000), # default: start date in july 2017 (Binance launch)
        'end_date': int(time.time()*1000) # default: actual time to get the up to date data
    }

    get_data = getData(**params)
    get_data.update_file_path(raw_data_path)
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
    


#################################################################################
#### [ ] Define Registered Users Methods
#################################################################################

@api.get('/verify_user', name = 'User login verification',
                         description = 'Verify if user is registered in the users database',
                         tags = ['Users'])
async def verify_user_route(username: str = Depends(verify_user)):
    """
    Returns confirmation is login protocol is succesfull
    """
    
    response = '{} is a registered user'.format(username)
    
    return response



#### [ ] Get the model parameters
@api.get('/prediction', name = 'Get prediction',
                         description = 'Get the prediction with the actual data. Return "BUY" or "WAIT".',
                         tags = ['Users'])
async def get_prediction(
                    asset: Asset,
                    interval: Intervals,
                    model_name: str = 'model_test',
                    username: str = Depends(verify_user)):
    """
    Returns confirmation is login protocol is succesfull
    """
    
    # read the dataset
    open_csv = openFile(raw_data_path, asset, interval)
    df = open_csv.data

    # preprocessing
    length=7
    factor=10

    # get the features
    X_transform = getFeatures(length, factor)
    X = X_transform.fit_transform(df.close)

    tm = tdbotModel(models_path, model_name)
    response = "BUY" if tm.get_prediction(X.iloc[-1:])==1 else "WAIT"
    
    return response


#################################################################################
#### [ ] Define Admins Methods
#################################################################################


from src.models.train_model import tdbotModel
from src.data.read_dataset import openFile
from src.features.build_features import getFeatures, getTarget

class Params(BaseModel):
    features_length: int = Field(default=7)
    features_factor: int = Field(default=10)
    target_ema_length: int = Field(default=7)
    target_diff_length: int = Field(default=14)
    target_pct_threshold: float = Field(default=0.2)
    model_n_neighbors: int = Field(default=30)
    model_weights: str = Field(default="uniform")
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
    params_checked["n_neighbors"] = params.model_n_neighbors if (type(params.model_n_neighbors)==int and params.model_n_neighbors<36) else 30
    params_checked["weights"] = params.model_weights if params.model_weights in ["uniform", "distance"] else "uniform"
    return params_checked

def get_params_cv(params):
    # function to update the input data if incorrect
    params_checked = {}
    params_checked["n_splits"] = params.cv_n_splits if type(params.cv_n_splits)==int else 5
    return params_checked

#### [ ] Get the model parameters
@api.get('/get_model_params', name = "Get the parameters of a model",
                         description = 'Verify the parameters actually in use for a given model (using the model_name as an ID)',
                         tags = ['Admins'])
async def get_model_params(model_name: str = 'model_test',
                           username: str = Depends(verify_admin)):
    """
    Returns the parameters of the model
    """

    tm = tdbotModel(models_path, model_name)
    response = tm.get_params()
    
    return response


#### [ ] Update the model parameters

@api.put('/update_model_params', name = "Create or Update the parameters of a model",
                         description = 'Create or Update the parameters for a model',
                         tags = ['Admins'])
async def update_model_params(params: Params,
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
    tm = tdbotModel(models_path, model_name)
    
    response = {}
    response['model_name'] = model_name
    response['previous_parameters'] = tm.get_params()
    response['response'] = tm.update_params(pp_params)
    response['new_parameters'] = tm.get_params()

    return response


#### [ ] assess a model 
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
    tm = tdbotModel(models_path, 'assess')
    
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


#### [ ] assess a model - with financial indicator
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
    tm = tdbotModel(models_path, 'assess')
    
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

#### [ ] train the model 
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
    df = open_csv.data

    # open a instance of tdbotModel
    tm = tdbotModel(models_path, model_name)

    # read the params
    params = tm.get_params()

    if params==False:
        response = "No existing parameters for this model. Use /update_model_params to create the parameters."
    else:
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