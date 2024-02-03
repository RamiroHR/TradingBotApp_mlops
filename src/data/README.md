# How to download the data

Use the file `make_dataset.py` to get the data from a specific pair.
The data will be downloaded using the Binance API.

```
python3 make_dataset [OPTIONS] RAW_DATA_FILEPATH
```
OPTIONS:
* --pair : trading pair you wish to download (format example: BTCUSDT)
* --interval : interval required (See below all the available intervals)

ARGUMENT:
* RAW_DATA_FILEPATH

### Intervals
The following interval are available in this app:<br>
`1s`, `1m`,  `5m`,  `15m`,  `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`
