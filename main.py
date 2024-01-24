from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

## instantiate the api and define the tags meatadata & security protocol
tags_metadata = [
    {'name': 'Public', 'description':'Methods available to everyone'},
    {'name': 'Users', 'description':'Methods available for registered user'},
    {'name': 'Admins', 'description':'Methods available for administrators'}
]


api = FastAPI(
    title = "Trading Bot App",
    description = "Enhance your profits and manage risks with AI assited investment strategies.",
    version = '1.0.0',
    openapi_tags = tags_metadata
)



#################################################################################
#### Login protocols 
#################################################################################
# [ ] need to be encrypted or from "secrets" !!


security = HTTPBasic()

## registered users credentials:
users_db = {
    "alice" : "wonderland",
    "bob" : "builder",
    "clementine": "mandarine"
}

## adminitrator credentials:
admins_db = {
    "admin" : "4dm1N"
}


## define functions to verify credentials: users & admins
def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username

    if not(users_db.get(username)) or not(credentials.password == users_db[username]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    print(username)

    if not(admins_db.get(username)) or not(credentials.password == admins_db[username]):
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
 



#################################################################################
#### [ ] Define Registered Users Methods
#################################################################################
@api.get('/verify_user', name = 'User login verification',
                         description = 'Verify if user is registered',
                         tags = ['Users'])
async def verify_user(username: str = Depends(verify_user)):
    """
    Returns confirmation is login protocol is succesfull
    """
    
    response = '{} is a registered user'.format(username)
    
    return {'status': response}
    


