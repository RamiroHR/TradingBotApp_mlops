from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# creating a FastAPI server
server = FastAPI(title='User/Admin database API')


# creating a connection to the MongoDB database
from pymongo import MongoClient

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

# Connect to MongoDB Atlas
# client = MongoClient(uri)

# creating a User class
class User(BaseModel):
    user_id: int = 0
    username: str = 'daniel'
    email: str = 'daniel@datascientest.com'


# test API
@server.get('/status')
async def get_status():
    """Returns 1
    """
    return 1

## test connection to the database
@server.get('/test-db-connection')
async def test_mysql_engine():
    """
    Test the correct creation of the MySQL engine with the recovered dtabase credentials
    """
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
        return {"status": "Pinged your deployment. You successfully connected to MongoDB Atlas!"}
    except Exception as e:
        # If there's an exception, display the error
        return {"error": str(e)}


## simple query: get all users
def get_all_users():
    with MongoClient(uri) as client:
        db = client.api_login_credentials
        collection = db.users
        cursor = collection.find()
        # Convert ObjectId to string for each document
        users = [{**user, "_id": str(user["_id"])} for user in cursor]
        return users
    
# Endpoint to retrieve all users
@server.get("/users")
async def retrieve_users():
    users = get_all_users()
    return users


# Parametrized query: get a particular user
@server.get("/users/{username}")
async def get_user_by_username(username: str):
    # Connect to MongoDB
    with MongoClient(uri) as client:
        # Access the database and collection
        db = client.api_login_credentials
        collection = db.users
        
        # Query MongoDB to find the user by username
        user = collection.find_one({"username": username})

        # If user not found, raise HTTPException with status code 404
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Convert ObjectId to string if necessary
        user["_id"] = str(user["_id"])

        return user
    
@server.get("/usernames")
async def get_usernames():
    # Connect to MongoDB
    with MongoClient(uri) as client:
        # Access the database and collection
        db = client.api_login_credentials
        collection = db.users
        
        # Query MongoDB to retrieve all usernames
        usernames = [user["username"] for user in collection.find()]  # Assuming username is a field in your collection

        return usernames


#############################################################
################  Microservice: verify user  ################

from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# Define security for Basic Authentication
security = HTTPBasic()

# Verify user function
def verify_user_credentials(credentials: HTTPBasicCredentials = Depends(security)):

    # user credentials to verify
    username = credentials.username
    password = credentials.password

    # Connect to MongoDB and check credentials
    with MongoClient(uri) as client:
        db = client.api_login_credentials
        collection = db.users
        user = collection.find_one({"username": username, "password": password})

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    return username

# Expose an endpoint to verify user credentials
@server.get("/verify-user")
async def verify_user_endpoint(credentials: HTTPBasicCredentials = Depends(verify_user_credentials)):
    return {"message": "User verified"}