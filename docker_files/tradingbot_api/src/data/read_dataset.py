
import os
import pandas as pd

#>>>
from pymongo import MongoClient

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
#<<<

class openFile:
    def __init__(self, data_path, asset, interval):
        self.data_path=os.path.abspath(data_path)
        self.asset = asset
        self.interval = interval
        self.data = self.open_csv()

##>>>
    # def open_csv(self):
    #     filename = self.asset+"-"+self.interval+"-raw.csv"
    #     file_path = os.path.join(self.data_path, filename)
    #     df = pd.read_csv(os.path.abspath(file_path))
    #     df['datetime']=pd.to_datetime(df["openT"], utc=True, unit="ms")
    #     df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
    #     return df


    def open_csv(self):
        coll_name = self.asset+"-"+self.interval+"-raw"
        
        with MongoClient(uri) as Mclient:
            db = Mclient.asset_price_hist            
            collection = db[coll_name]
            cursor = collection.find()
                
            df = pd.DataFrame(list(cursor))
        
        if len(df)>0:
            df.drop(columns=['_id'], inplace=True)
            df['datetime']=pd.to_datetime(df["openT"], utc=True, unit="ms")
            df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        
        return df
##<<<

    def is_data(self):
        if len(self.data) <= 0:
            return False
        else:
            return True