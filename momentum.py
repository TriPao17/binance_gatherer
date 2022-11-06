from tools import BinanceCustomClient

# Instantiate Simple API
api = BinanceCustomClient()

# Get Symbols
data = api.get_symbols(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                       start_str="2022-08-20",
                       end_str="2022-09-10",
                       interval="1-hour")

