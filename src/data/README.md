# How to download the data

Use the file `make_dataset.py` to get the data from a specific pair.
The data will be downloaded using the Binance API.
<br>
This py file can be used either in command line or called by another script.

## Using command line


```
python3 make_dataset [OPTIONS] RAW_DATA_FILEPATH
```
OPTIONS:
* --pair : trading pair you wish to download (format example: BTCUSDT)
* --interval : interval required (See below all the available intervals)

ARGUMENT:
* RAW_DATA_FILEPATH

## Using another script

You can call this script.

```
from make_dataset import getData

 # set the parameters
params = {
    'pair': pair, # pair to be dowloaded (ex.: BTCUSDT)
    'interval': interval, # interval required (ex.: 1d). See below the available interval.
    'pitch': 1000, # number of lines per call
    'start_date': int(datetime(year=2017, month=1, day=1, hour=0, minute=0, tzinfo=timezone.utc).timestamp()*1000), # minimum date where you want to start collecting the data
    'end_date': int(time.time()*1000) # finish date
}

get_data = getData(**params) # Create an instance of getData 
get_data.update_file_path("raw_data") # define the path of the folder
get_data.getKlines() # get all the klines and record into the right folder
```

## Intervals
The following interval are available in this app:<br>
`1s`, `1m`,  `5m`,  `15m`,  `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`
