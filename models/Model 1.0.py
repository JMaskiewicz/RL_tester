# import libraries
import numpy as np
import pandas as pd
import datetime
import math
import csv
from datetime import datetime
from datetime import timedelta
import pandas_ta as ta
import random


import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# functions
from functions.loading_data import load_data
from functions.functions import add_indicators, get_week_indices, split_data_by_weeks
from functions.strategy import apply_strategy
from functions.metrics import generate_report, extract_trades, plot_trades

random.seed(1234)
# Set the random seed
seed_value = 1234
np.random.seed(seed_value)

# code
RSI_length = 3
EMA_length = 21
df = load_data(currencies=['EURUSD'], timestamp_x='5min')
df = add_indicators(df, currency='EURUSD', RSI_length=RSI_length, EMA_length=EMA_length)
df = df.dropna()
df = df['2022-08-01':]

# splits = split_data_by_weeks(df, 104, 26, 26, 13)  # 52 weeks = 1 year
#splits = split_data_by_wktorych
# eeks(df, 20, 26, 26, 13)  # 52 weeks = 1 year
'''
for train, test, val in splits:
    print("Train:", train.index.min(), train.index.max())
    print("Test:", test.index.min(), test.index.max())
    print("Validation:", val.index.min(), val.index.max())
    print("----------------------")
'''

currency = 'EURUSD'
RSI_breakout = 30
EMA_breakout = 21
hold_period = 10

# Extract the DataFrames from the tuple
#train_df, test_df, val_df = splits[0]

# Apply the strategy to the train_df
df = apply_strategy(df=df,
                    currency=currency,
                    RSI_breakout=RSI_breakout,
                    EMA_breakout=EMA_breakout,
                    RSI_length=RSI_length,
                    EMA_length=EMA_length,
                    hold_period=hold_period)

# calculate metrics
INITIAL_CAPITAL = 10000
TRANSACTION_COST_PCT = 0 #0.001  # for example, 0.1% transaction cost

trades = extract_trades(df, currency)
trades = pd.DataFrame(trades)

report = generate_report(df, currency, INITIAL_CAPITAL, TRANSACTION_COST_PCT)

for key, value in report.items():
    print(f"{key:<30} {value}")

# Replace the old tuple in the splits list with the new tuple
#splits[0] = (train_df, test_df, val_df)


