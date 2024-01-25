import pandas as pd
#import numpy as np

db_path = "./data/all_users/database.json"

################################################################################
### Import existing database ---- JSON file
################################################################################

import json

def get_db(dbtype):
    with open(db_path) as json_file:
        data = json.load(json_file)   # load into a dict
        
        data_dbtype = data[dbtype]
        
        return data_dbtype       

def get_registered_users():
    return get_db(dbtype = 'users_db')

def get_admin_users():
    return get_db(dbtype = 'admin_db')

def write_database(new_user, filename = db_path):
    with open(filename, 'r+') as file:
        db_data = json.load(file)
        
        db_data["users_db"].update(new_user)
        
        file.seek(0)
        
        json.dump(db_data, file, indent = 4)
        
        
    