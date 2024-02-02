# Creates a users-admins database at the specified location by the path variable
# This dataset is only for the development phase of the API source code.

import os
import json 

def create_all_users_database():
    
    database_path = './data/all_users/'
    database_name = 'database.json'

    filepath = database_path + database_name

    # create all intermeadiate folders is they do not exist.
    os.makedirs(database_path, exist_ok=True)
    
    data = {
        "users_db": {
            "user_1": {
                "username": "user_1",
                "password": "u_one"
            },
            "user_2": {
                "username": "user_2",
                "password": "u_two"
            },
            "user_3": {
                "username": "user_3",
                "password": "u_three"
            }
        },
        "admin_db": {
            "admin_1": {
                "username": "admin_1",
                "password": "a_one"
            },
            "admin_2": {
                "username": "admin_2",
                "password": "a_two"
            }
        }
    }

    with open( filepath, "w") as file: 
        json.dump(data, file, indent = 4 ) 