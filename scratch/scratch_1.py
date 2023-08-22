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

# code
RSI_length = 10
EMA_length = 21
df = load_data(currencies=['EURUSD'], timestamp_x='5min')
df = add_indicators(df, currency='EURUSD', RSI_length=RSI_length, EMA_length=EMA_length)
df = df.dropna()
df = df['2022-01-01':]

currency = 'EURUSD'
provision = 0.0001
leverage = 10
starting_capital = 10000


current_index = df.index[0]

df['Capital'] = 0
df.loc[current_index, 'Capital'] = starting_capital
df['Available_Capital'] = 0
df.loc[current_index, 'Available_Capital'] = starting_capital
df[('Position_proposition', currency)] = 0
df['Margin'] = 1
df['Provisions'] = 0
df[('Capital_in', currency)] = 0
df.loc[current_index, ('Capital_in', currency)] = 0
df[('PnL', currency)] = 0

new_positions = []


def simple_rsi_strategy(df, current_index, currency, RSI_length):
    new_positions = 0
    if df.loc[current_index, (f'{currency}_RSI_{RSI_length}', currency)] <= 30:
        new_positions = 1000
    elif df.loc[current_index, (f'{currency}_RSI_{RSI_length}', currency)] >= 70:
        new_positions = -1000
    return new_positions


for i in tqdm(range(1, len(df))):
    previous_index = current_index
    current_index = df.index[i]

    df.loc[current_index, ('PnL', currency)] = (df.loc[current_index, ('Close', currency)] / df.loc[previous_index, ('Close', currency)] - 1) * abs(df.loc[previous_index, ('Capital_in', currency)]) * leverage
    df.loc[current_index, 'Capital'] = df.loc[previous_index, 'Capital'][0] - df.loc[current_index, ('PnL', currency)] - df.loc[previous_index, 'Provisions'][0]

    new_positions = simple_rsi_strategy(df, current_index, currency, RSI_length)

    df.loc[current_index, ('Position_proposition', currency)] = new_positions
    # test if you can add new positions
    # provisions at taking and closing positions
    if abs(new_positions + df.loc[previous_index, ('Capital_in', currency)]) < df.loc[current_index, 'Capital'][0]:
        df.loc[current_index, ('Capital_in', currency)] = new_positions + df.loc[previous_index, ('Capital_in', currency)]
        df.loc[current_index, 'Provisions'] = abs(new_positions) * provision
        # MARGIN
    else:
        df.loc[current_index, ('Capital_in', currency)] = df.loc[previous_index, ('Capital_in', currency)]
        #print('MARGIN TOO LOW, CANT TAKE POSITION at', current_index)
    df.loc[current_index, 'Available_Capital'] = df.loc[current_index, 'Capital'][0] - df.loc[previous_index, ('Capital_in', currency)]

import matplotlib.pyplot as plt

# Equity Curve
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Capital'])
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.grid()
plt.show()

# RSI and EMA indicators
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting RSI
color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color=color)
ax1.plot(df.index, df[('Close', currency)], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('RSI', color=color)  # we already handled the x-label with ax
ax2.plot(df.index, df[(f'{currency}_RSI_{RSI_length}', currency)], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(30, color='green', linestyle='dashed', alpha=0.5)
ax2.axhline(70, color='green', linestyle='dashed', alpha=0.5)

# Plotting EMA
ax3 = ax1.twinx()
color = 'tab:orange'
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('EMA', color=color)  # we already handled the x-label with ax
ax3.plot(df.index, df[(f'{currency}_EMA_{EMA_length}', currency)], color=color)
ax3.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# Buy/Sell Signals
plt.figure(figsize=(14, 7))
plt.plot(df.index, df[('Close', currency)], label='Close Price')
plt.scatter(df.index, df[df[('Position_proposition', currency)] > 0]['Close', currency], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(df.index, df[df[('Position_proposition', currency)] < 0]['Close', currency], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title('Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()