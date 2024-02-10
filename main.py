from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from enum import Enum
from src.data.make_dataset import getData
from datetime import datetime, timezone
import time

import database_management_tools as dbm

## instantiate the api and define the tags meatadata & security protocol
tags_metadata = [
    {'name': 'Public', 'description':'Methods available to everyone'},
    {'name': 'Users', 'description':'Methods available for registered user'},
    {'name': 'Admins', 'description':'Methods available for administrators'}
]


api = FastAPI(
    title = "Trading Bot App",
    description = "Enhance profits and manage risks with AI assited investment strategies.",
    version = '1.0.0',
    openapi_tags = tags_metadata
)


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
allowed_interval = ['4h', '1d', '1w']
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
    get_data.update_file_path("data/raw")
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



#################################################################################
#### [ ] Define Admins Methods
#################################################################################

