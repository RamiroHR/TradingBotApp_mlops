from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

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


security = HTTPBasic()

users_db = dbm.get_registered_users()
admins_db = dbm.get_admin_users()

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

@api.get('/app_status', name = 'Quick test if the API is running',
                    tags = ['Public'])
async def get_app_status():
    return {'status' : 'App is available'}


class User(BaseModel):
    username: str
    password: str
 
@api.post('/sign_up', name = 'Create new user',
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
     


#################################################################################
#### [ ] Define Registered Users Methods
#################################################################################

@api.get('/verify_user', name = 'User login verification',
                         description = 'Verify if user is registered',
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
