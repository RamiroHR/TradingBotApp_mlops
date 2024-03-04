
import os
import pandas as pd

class openFile:
    def __init__(self, data_path, asset, interval):
        self.data_path=os.path.abspath(data_path)
        self.asset = asset
        self.interval = interval
        self.data = self.open_csv()


    def open_csv(self):
        filename = self.asset+"-"+self.interval+"-raw.csv"
        file_path = os.path.join(self.data_path, filename)

        try:
            df = pd.read_csv(os.path.abspath(file_path))
            df['datetime']=pd.to_datetime(df["openT"], utc=True, unit="ms")
            df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
        except:
            df = pd.DataFrame()

        return df

    def is_data(self):
        if len(self.data) <= 0:
            return False
        else:
            return True