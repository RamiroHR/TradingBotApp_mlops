# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# import libraries for Data download
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import os
from binance.spot import Spot # pip install binance-connector-python

#>>>
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
#<<<

class getData:
    """
    get the data from binance on a specific pair
    """
    def __init__(self, pair='BTCUSDT', interval='1m', pitch=500, start_date='', end_date=''):
        # setup basic variables
        self.pair=pair # pair to scrap. Example: BTCUSDT
        self.interval=interval # Interval of time to consider: 1s, 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d
        self.pitch=pitch # Number of points to collect at each call
        self.pitch_ms=self.pitch_to_ms() # Number of ms composing the duration (interval * pitch)

        # setup the start and end dates (to be updated by the user depending on the needs)
        self.startDate=self.setDate(year=2023, month=1, day=1, hour=0, min=0) if start_date=='' else start_date # set the default start date
        self.endDate=self.setDate(year=2023, month=1, day=1, hour=2, min=0) if end_date=='' else end_date # set the default end date

#>>>
        # # setup the path to record the data
        # self.update_file_path("./")
#<<<
        # setup the api call
        self.base_url = "https://api4.binance.com"
        self.client=Spot(self.base_url)

        # colums names
        self.columns_name = ['openT',
           'open', 'high', 'low', 'close', 'baseVol', 'closeT','quoteVol','nbTrade',
           'takerBaseVol', 'takerQuoteVol', '0']

#>>>
    # def update_file_path(self, file_path):
    #     self.file_path=os.path.abspath(file_path) # convert the directory into an absolute directory in order to avoid potential bugs during deployments
    #     # Check if the folder exists
    #     if not os.path.exists(self.file_path):
    #         # If it doesn't exist, create it
    #         os.makedirs(self.file_path)
    #         print(f"Directory '{self.file_path}' created successfully.")
    #     else:
    #         print(f"Directory '{self.file_path}' already exists.")
#<<<

    def setDate(self, year=2023, month=1, day=1, hour=0, min=0):
        """
        This function returns the timestamp in ms and within the timezone utc
        """
        return int(datetime(year, month, day, hour, min, tzinfo=timezone.utc).timestamp()*1000)

    def pitch_to_ms(self):
        """
        This function return the number of ms equivalent to the (pitch * interval)
        """
        # dict with the intervals fiven in number of seconds
        interval_to_sec = {
            '1s':1, 
            '1m':60,
            '5m':60*5,
            '15m':60*15,
            '30m':60*30, 
            '1h':60*60, 
            '2h':60*60*2, 
            '4h':60*60*4, 
            '6h':60*60*6, 
            '8h':60*60*8, 
            '12h':60*60*12, 
            '1d':86400, 
            '3d':86400*3, 
            '1w':86400*7
        }
        try:
            #return msdatetime+self.pitch*interval_to_sec[self.interval]*1000
            return self.pitch*interval_to_sec[self.interval]*1000
        except:
            print("Error with the pitch_to_ms function.")

    def getKlines(self, file_name=''):
        """
        This is the main function to be used by the user to scrap the binance data for a ticker.
        It will get the data from a defined folder. If some historical data exists, it will update the data. If not, it will scrap the bull historical.
        """


        # create the file name
        file_name = self.pair+'-'+self.interval+'-raw.csv' if file_name=='' else file_name
#>>>        
        # if os.path.isfile(os.path.join(self.file_path, file_name)):
        #     # if the file already exists, open it
        #     df_temp = pd.read_csv(os.path.join(self.file_path, file_name))

        #     # take the last entry as a start date for the update
        #     self.startDate = df_temp.iloc[-1,0]
        #     print('Data will be updated stating from the following timestamp (ms) : ', self.startDate, ' (index:', df_temp.index[-1],')')
        # else:
        #     df_temp = None
        #     print('There is no file to update')
#<<<

#>>>
        # create the collection name (without the '.csv' name part)
        coll_name = self.pair+'-'+self.interval+'-raw' if file_name=='' else file_name[:-4]

        with MongoClient(uri) as Mclient:
            db = Mclient.asset_price_hist            
            
            if coll_name in db.list_collection_names():
                # if collection file already exists, pull it
                collection = db[coll_name]
                cursor = collection.find()
                
                df_temp = pd.DataFrame(list(cursor))
                
                # take the last entry as a start date for the update
                self.startDate = df_temp.iloc[-1,0]
                print('Data will be updated stating from the following timestamp (ms) : ', self.startDate, ' (index:', df_temp.index[-1],')')
            else:
                df_temp = None
                print('There is no file to update')
#<<<


        # get the klines
        df = self.updateKlines()

        # if df_temp exists, concat df_temp and df
        if isinstance(df_temp, pd.DataFrame):
            df = pd.concat([df, df_temp], axis=0, ignore_index=True)

        # clean the data
        df = self.cleanKlines(df)

#>>>
        # # export to csv
        # df.to_csv(os.path.join(self.file_path, file_name), index=False)
        # print('Data updated')
#<<<

#>>>
        ## >>>>>>>>>>  export to MongoDB  <<<<<<<<<<<<<<< 
        data = df.to_dict(orient='records')
        
        with MongoClient(uri) as Mclient:
            db = Mclient.asset_price_hist
            collection = db[coll_name]
            # collection.drop()
            collection.insert_many(data)
        print('Data updated')
        ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#<<<
        return df
    

    def updateKlines(self):
        """
        This function collect all the data from a startDate to endDate
        """
        klines = []
        start = self.startDate
        while start < self.endDate:
            # Get klines of the pair for the defined interval
            klines = klines + self.client.klines(self.pair, self.interval, startTime=start, limit=self.pitch)
            start += self.pitch_ms

        # transformation to dataframe
        df = pd.DataFrame(klines, columns=self.columns_name)

        # removal of the last line for which the data might not be frozen (price still moving)
        df = df.iloc[:-1]

        # removal of the last column which is a null column
        df.drop(df.columns[-1], axis=1, inplace=True)

        # update the dtypes to numeric
        df = df.apply(pd.to_numeric)

        # clean the data
        df = self.cleanKlines(df)

        # return the dataframe
        return df

    def cleanKlines(self, data):
        """
        This function "clean" the data in a consistent way
        """
        return data.sort_values(by='openT').drop_duplicates().reset_index(drop=True)


@click.command()
@click.argument('raw_data_filepath', type=click.Path())
#@click.argument('output_filepath', type=click.Path())
@click.option('--pair', default='BTCUSDT', help='Specify a trading pair (default: BTCUSDT).')
@click.option('--interval', default='1d', help='Specify a trading pair (default: BTCUSDT).')
def main(raw_data_filepath, pair, interval):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # set the parameters
    params = {
        'pair': pair,
        'interval': interval,
        'pitch': 1000,
        'start_date': int(datetime(year=2017, month=1, day=1, hour=0, minute=0, tzinfo=timezone.utc).timestamp()*1000),
        'end_date': int(time.time()*1000)
    }

    get_data = getData(**params)
    get_data.update_file_path(raw_data_filepath)
    get_data.getKlines()

    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    
