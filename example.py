from tools import BinanceCustomClient

# Instantiate Simple API
api = BinanceCustomClient()

# Get Data
data = api.get_symbols(symbols=["BTCUSDT", "ETHUSDT", "NEOUSDT"],
                       start_str="2021-01-01",
                       end_str="2022-08-17",
                       interval="1-hour",
                       verbose=True)
