# description
"""

"""

# import libraries
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import os
from tqdm import tqdm

data_folder = "./data"


def load_data(currencies, timestamp_x):
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    dfs = []
    for currency in tqdm(currencies):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_folder = os.path.join(project_root, 'data_sets')
        file_path = os.path.join(data_folder, f'{currency}_2003-2022.txt')
        df = pd.read_csv(file_path, sep=',')
        df['Date'] = df['Date'] + " " + df['time']
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
        df = df.drop(["time"], axis=1)
        df.set_index('Date', inplace=True)
        # Convert the naive datetime index to UTC and then to Warsaw time
        # df.index = df.index.tz_localize('America/New_York').tz_convert('Europe/Warsaw')
        df = df[['Open', 'High', 'Low', 'Close']]
        df = df.resample(timestamp_x).agg(agg_dict)
        df = df.dropna()
        df['Currency'] = currency
        df.reset_index(inplace=True)
        df.set_index(['Date', 'Currency'], inplace=True)
        df = df.unstack('Currency')
        dfs.append(df)
    df_all = pd.concat(dfs, axis=1)
    return df_all
