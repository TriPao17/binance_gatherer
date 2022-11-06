import datetime
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta as td, datetime as dt
from functools import reduce
from typing import Tuple, List

import pandas as pd
from binance import Client


# todo create a consistancy check before delivering data

class BinanceCustomClient:
    # Instantiation Function
    def __init__(self, api_key_path: str = "keys/api_key.key", api_secret_path: str = "keys/api_secret.key"):
        # Save Paths
        self.api_key_path = api_key_path
        self.api_secret_path = api_secret_path

        # Load Keys
        api_key, secret_key = self._load_keys()

        # Instantiate Client
        self.client = Client(api_key, secret_key)

        # Check if data folder exists, if not create it
        self._create_folder("data")

        # Instantiate Structure Reader
        self.reader = StructureReader()

    # Load Keys from Files
    def _load_keys(self) -> Tuple[str, str]:
        # Load Api Key
        with open(self.api_key_path, 'r') as ak:
            api_key = ak.read()
        # Load Api Secret
        with open(self.api_secret_path, 'r') as sk:
            api_secret = sk.read()
        # Return Api Key and Api Secret as a Tuple
        return api_key, api_secret

    @staticmethod
    def _create_folder(folder_path: str) -> None:
        if os.path.isdir(folder_path) is False:
            os.mkdir(folder_path)

    @staticmethod
    def _write_pickle(obj: object, path: str):
        with open(path, 'wb') as pickle_saver:
            pickle.dump(obj, pickle_saver)

    @staticmethod
    def read_pickle(path: str):
        with open(path, 'rb') as pickle_loader:
            return pickle.load(pickle_loader)

    @staticmethod
    def increase_date(dt_date: datetime.datetime):
        return dt_date + td(days=1)

    @staticmethod
    def get_days_between_bounds(date_start: str, date_end: str) -> List[str]:
        """Creates a list of date including the first and the last date from a range of dates.

        :param date_start: First date in the range of dates.
        :param date_end: Last date in the range of dates.
        :return: List of dates between date_start and date_end
        """

        # Convert Dates
        date_start = dt.strptime(date_start, "%Y-%m-%d").date()
        date_end = dt.strptime(date_end, "%Y-%m-%d").date()

        # Get Delta
        delta = date_end - date_start

        # Get All Days
        days = [str(date_start + td(days=i)) for i in range(delta.days + 1)]

        # Return Days
        return days

    # Get Historical
    def get_klines(self,
                   symbol: str,
                   start_str: str,
                   end_str: str,
                   interval: str) -> pd.DataFrame:

        # Convert Dates to datetime objects
        dt_start = datetime.datetime.strptime(start_str, "%Y-%m-%d")
        dt_end = datetime.datetime.strptime(end_str, "%Y-%m-%d")

        # Chek if dates are sequential
        if dt_end < dt_start:
            raise ValueError("The end date must be superior or equal to the start date")

        # Chek if end date in not in the future
        if dt_end > datetime.datetime.now():
            raise ValueError("The end date can't be in the future")

        # Create filename
        _filename = f"{symbol}_{interval}_{start_str}_{end_str}.pkl"

        # Create filepath
        _filepath = f"data/{interval}/{_filename}"

        # Check if file exists, if it does load and return it
        if os.path.exists(_filepath):
            return self.read_pickle(_filepath)

        # Increase date by one day, so that we can include the ending in the data
        dt_end = self.increase_date(dt_end)
        end_str = datetime.datetime.strftime(dt_end, "%Y-%m-%d")

        # Choose Interval
        if interval == "1-month":
            _interval = Client.KLINE_INTERVAL_1MONTH

        elif interval == "1-week":
            _interval = Client.KLINE_INTERVAL_1WEEK

        elif interval == "1-day":
            _interval = Client.KLINE_INTERVAL_1DAY
        elif interval == "3-day":
            _interval = Client.KLINE_INTERVAL_3DAY

        elif interval == "1-hour":
            _interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == "2-hour":
            _interval = Client.KLINE_INTERVAL_2HOUR
        elif interval == "4-hour":
            _interval = Client.KLINE_INTERVAL_4HOUR
        elif interval == "6-hour":
            _interval = Client.KLINE_INTERVAL_6HOUR
        elif interval == "8-hour":
            _interval = Client.KLINE_INTERVAL_8HOUR
        elif interval == "12-hour":
            _interval = Client.KLINE_INTERVAL_12HOUR

        elif interval == "1-minute":
            _interval = Client.KLINE_INTERVAL_1MINUTE
        elif interval == "3-minute":
            _interval = Client.KLINE_INTERVAL_3MINUTE
        elif interval == "5-minute":
            _interval = Client.KLINE_INTERVAL_5MINUTE
        elif interval == "15-minute":
            _interval = Client.KLINE_INTERVAL_15MINUTE
        elif interval == "30-minute":
            _interval = Client.KLINE_INTERVAL_30MINUTE

        else:
            raise ValueError(f"The interval given {interval} is invalid.\n"
                             f"Interval must one of: 1-month, 1-week, 1-day, 3-day, 1-hour, 2-hour, 4-hour, 6-hour\n"
                             f"8-hour, 12-hour, 1-minute, 3-minute, 5-minute, 15-minute, 30-minute")

        # Using the base client to collect raw data
        raw_data = self.client.get_historical_klines(symbol=symbol,
                                                     interval=_interval,
                                                     start_str=start_str,
                                                     end_str=end_str)

        # Converting raw data into a DataFrame Object
        data = pd.DataFrame(raw_data)

        # Add Columns Names
        data.columns = ['open_time',
                        (symbol, 'open'),
                        (symbol, 'high'),
                        (symbol, 'low'),
                        (symbol, 'close'),
                        (symbol, 'volume'),
                        'close_time',
                        (symbol, 'quote_asset_volume'),
                        (symbol, 'n_trades'),
                        (symbol, 'taker_buy_base_asset_volume'),
                        (symbol, 'taker_buy_quote_asset_volume'),
                        (symbol, 'can_be_ignored')]

        # Transform Time Columns to Date Time
        data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
        data["close_time"] = pd.to_datetime(data["close_time"], unit="ms")

        # Move Close Time column to second position
        _close_time = data.pop("close_time")
        data.insert(1, "close_time", _close_time)

        # Transforming Columns to Float
        float_cols = [(symbol, 'open'),
                      (symbol, 'high'),
                      (symbol, 'low'),
                      (symbol, 'close'),
                      (symbol, 'volume'),
                      (symbol, 'quote_asset_volume'),
                      (symbol, 'taker_buy_base_asset_volume'),
                      (symbol, 'taker_buy_quote_asset_volume'),
                      (symbol, 'can_be_ignored')]
        data[float_cols] = data[float_cols].astype(float)

        # Trim edges
        data = data[(data["open_time"] >= dt_start) & (data["open_time"] < dt_end)]

        # Check if interval folder exists, if not create it
        self._create_folder(f"data/{interval}")

        # Save File
        self._write_pickle(obj=data, path=_filepath)

        # Return Cleaned Data
        return data

    def get_symbol(self, symbol: str, interval: str, start_str: str, end_str: str,
                   verbose: bool = True) -> pd.DataFrame:
        # Get all dates necessary
        dates_req = self.get_days_between_bounds(start_str, end_str)

        # Get files that we have
        df_files = self.reader.files_properties()

        # Filter on relevant files for request
        df_files_relevant = df_files[(df_files["symbol"] == symbol) & (df_files["interval"] == interval)]

        # Get Stored Dates & Data
        available_dates = []
        available_data = []

        # Loop over relevant files
        for row in df_files_relevant.iterrows():
            # Get List of available dates for each file and concatenate them
            available_dates += self.get_days_between_bounds(row[1]["date_start"], row[1]["date_end"])
            # Read the files that we have, and store them in the list available_data
            available_data.append(BinanceCustomClient.read_pickle(row[1]["path"]))
        # Clean out any duplicate dates
        available_dates = list(set(available_dates))

        # Create Dataframes for both the requested and available dates
        df_dates_req = pd.DataFrame(dates_req, columns=["dates_req"])
        df_dates_ava = pd.DataFrame(available_dates, columns=["dates_data"])

        # Merge the two created frames together
        df_dates_merged = pd.merge(df_dates_req, df_dates_ava, left_on="dates_req", right_on="dates_data", how="left")

        # Extract the requirement dates list
        dates_ava_extracted = df_dates_merged["dates_data"].tolist()

        # Extract the available dates list, it will now contain nan elements for the missing dates in local data
        dates_req_extracted = df_dates_merged["dates_req"].tolist()

        # Loop over dates to find bounds of requests to perform
        pairs = []
        _pair = []
        for k in range(len(dates_ava_extracted)):
            # Date From requirements columns
            real_date = dates_req_extracted[k]

            # Check to see if iterating on the last Date
            is_last_date = True if k == len(dates_ava_extracted) - 1 else False

            # Next Date From Merged Column
            next_date_extracted = dates_ava_extracted[k + 1] if is_last_date is False else real_date

            # Current Date from real merged column
            date = dates_ava_extracted[k]

            # Check if current date is nan and is the first item of pair
            if type(date) is float and math.isnan(date) and len(_pair) == 0:
                _pair.append(real_date)

                # If we are on the last date it also is the closing date
                if is_last_date is True:
                    _pair.append(real_date)
                    pairs.append(_pair)

            # If the next date is not na, close the pair
            if type(next_date_extracted) is str and len(_pair) == 1:
                _pair.append(real_date)
                pairs.append(_pair)
                _pair = []

        # Print pairs to query
        pairs_len = len(pairs)

        # Request data according to pairs found
        requested_data = []
        for i, pair in enumerate(pairs):
            if verbose is True:
                print(f"Querying {symbol} {interval} from {pair[0]} to {pair[1]}. Query ({i + 1}/{pairs_len}).")
            requested_data.append(self.get_klines(symbol=symbol,
                                                  interval=interval,
                                                  start_str=pair[0],
                                                  end_str=pair[1]))

        # Merge Available and queried Fragments
        all_data = requested_data + available_data

        # Concatenate DataFrames
        out_data = pd.concat(all_data)

        # Remove duplicate rows
        out_data = out_data.drop_duplicates()

        # Convert str_times to datetime objects
        date_start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
        date_end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d") + td(days=1)

        # Trim the data according to start and end dates
        out_data = out_data[(out_data["open_time"] >= date_start_dt) & (out_data["open_time"] < date_end_dt)]

        # Sort Values and reset index
        out_data = out_data.sort_values("open_time", ascending=True).reset_index(drop=True)

        # Return Data
        return out_data

    def get_symbols(self, symbols: List[str], interval: str, start_str: str, end_str: str) -> pd.DataFrame:
        # Loop over symbols
        _symbols_len = len(symbols)
        _symbols_data = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for i, symbol in enumerate(symbols):
                _future = executor.submit(self.get_symbol,
                                          symbol=symbol,
                                          interval=interval,
                                          start_str=start_str,
                                          end_str=end_str,
                                          verbose=True)
                futures.append(_future)

            # Try to get results
            for future in futures:
                try:
                    _result = future.result()
                except ValueError:
                    pass
                _symbols_data.append(_result)

        # Merged data together
        merged_data = reduce(lambda left, right: pd.merge(left, right,
                                                          on=['open_time', 'close_time'],
                                                          how='outer'), _symbols_data)

        # Return Merged Data
        return merged_data


class StructureReader:
    def __init__(self, base_folder: str = "data"):
        self.base_folder = base_folder
        self.files_list = self.index_files()
        self.files_df = self.files_properties()

    def index_files(self) -> List[str]:
        _indexed_files = []
        # Loop Over Directories
        for root, dirs, files in os.walk(self.base_folder):
            for file_ in files:
                # Skip files that start with a dot
                if file_[0] == ".":
                    continue
                # Skip Roots that contain hidden directories
                elif "\\." in root:
                    continue
                _indexed_files.append(os.path.join(root, file_))

        return _indexed_files

    def files_properties(self) -> pd.DataFrame:
        files_list = self.index_files()
        split_tree = [path.split("\\") for path in files_list]
        files_df = pd.DataFrame(data=split_tree, columns=["base_folder", "interval_folder", "file"])

        # Handle case in which no data has ever been downloaded before
        if files_df.empty is True:
            empty_df = pd.DataFrame(columns=["base_folder", "interval_folder", "file", "symbol", "interval",
                                             "date_start", "date_end"])
            return empty_df

        files_df["path"] = files_list

        names_no_extension = files_df["file"].str.replace(".pkl", "", regex=False)
        df_properties = names_no_extension.str.split("_", expand=True)
        properties_cols = ["symbol", "interval", "date_start", "date_end"]
        df_properties.columns = properties_cols

        files_df[properties_cols] = df_properties
        return files_df
