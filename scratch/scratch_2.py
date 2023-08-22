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


def backtest(df, strategy, starting_capital=10000, provision=0.0001, leverage=10,
             currencies=['EURUSD', 'USDJPY', 'EURJPY'], **kwargs):
    # Setup DataFrame
    current_index = df.index[0]
    df['Capital'] = 0
    df.loc[current_index, 'Capital'] = starting_capital
    df['Available_Capital'] = 0
    df.loc[current_index, 'Available_Capital'] = starting_capital
    df['Margin'] = 1
    df['Provisions'] = 0

    for currency in currencies:
        df[('Position_proposition', currency)] = 0
        df[('Capital_in', currency)] = 0
        df.loc[current_index, ('Capital_in', currency)] = 0
        df[('PnL', currency)] = 0

    # Backtesting Loop
    for i in tqdm(range(1, len(df))):
        previous_index = current_index
        current_index = df.index[i]

        # Vectorized calculation of required capital
        capital_in_previous = df.loc[previous_index, ('Capital_in', currencies)]
        df.loc[current_index, 'Available_Capital'] = df.loc[previous_index, 'Capital'][0] - abs(
            capital_in_previous).sum()
        avaible_capital = df.loc[current_index, 'Available_Capital']

        # Get new positions from strategy function
        new_positions = strategy(df, current_index, currencies, **kwargs)  # , avaible_capital

        new_positions_series = pd.Series(new_positions).reindex(capital_in_previous.index.get_level_values(1))
        required_capital_series = abs(new_positions_series + capital_in_previous)
        total_required_capital = required_capital_series.sum()

        # Vectorized calculation of provisions
        new_provisions_series = pd.Series(new_positions).reindex(capital_in_previous.index.get_level_values(1))
        provisions_series = abs(new_provisions_series) * provision
        total_provisions = provisions_series.sum()

        # 1. Get the Close prices at the current and previous indexes for all currencies
        current_close_prices = df.loc[current_index, ('Close', currencies)]
        previous_close_prices = df.loc[previous_index, ('Close', currencies)]

        # 2. Get the Capital_in values at the previous index for all currencies
        previous_capital_in = df.loc[previous_index, ('Capital_in', currencies)]

        # 3. Calculate the returns for each currency
        current_returns = ((current_close_prices / previous_close_prices) - 1)

        # 4. Reset the MultiIndex of current_returns to match the MultiIndex of previous_capital_in
        current_returns.index = previous_capital_in.index

        # 5. Calculate the PnL series
        PnL_series = current_returns * previous_capital_in * leverage

        # Sum the PnL for all currencies to get the total PnL
        total_PnL = PnL_series.loc['Capital_in'].sum()

        # Update the dataframe with the calculated PnL for each currency
        df.loc[current_index, ('PnL', currencies)] = PnL_series.loc['Capital_in'].values

        # Update the dataframe with the new positions for each currency
        df.loc[current_index, ('Position_proposition', currencies)] = pd.Series(new_positions).values

        if total_required_capital <= df.loc[previous_index, 'Capital'][0]:
            df.loc[current_index, 'Provisions'] = total_provisions
            # Vectorized update of Capital_in for each currency
            df.loc[current_index, ('Capital_in', currencies)] = new_positions_series + capital_in_previous
        else:
            # Calculate the net change in positions
            net_change_positions = abs(capital_in_previous + new_positions_series) - abs(capital_in_previous)

            # Flatten the multi-index to a single level
            net_change_positions_flat = net_change_positions.reset_index(level=0, drop=True)

            # Reindex the series to match each other's indices
            net_change_positions_flat = net_change_positions_flat.reindex(new_positions_series.index)

            # Get the positions where net_change_positions > 0
            closing_positions = new_positions_series.where(net_change_positions_flat < 0, 0)

            # for cases where we have 1000 long, and we want to add 2000 short we can only short 1000 if there is no capital avaible
            smaller_abs_values = pd.Series(
                np.minimum(np.abs(closing_positions.values), np.abs(capital_in_previous.values)),
                index=closing_positions.index) * np.sign(new_positions_series)

            # Calculate the total capital from closing positions
            total_closing_capital = abs(smaller_abs_values).sum()

            # Vectorized update of Capital_in for each currency
            df.loc[current_index, ('Capital_in', currencies)] = capital_in_previous + smaller_abs_values

            if total_closing_capital > 0:
                # Calculate the total provisions
                total_provisions = abs(closing_positions).sum() * provision
                df.loc[current_index, 'Provisions'] = total_provisions
            else:
                total_provisions = 0

        df.loc[current_index, 'Capital'] = df.loc[previous_index, 'Capital'][0] - total_provisions + total_PnL

    return df


def simple_rsi_strategy(df, current_index, currencies, RSI_length):  # , avaible_capital, current_positions
    new_positions = {}
    for currency in currencies:
        if df.loc[current_index, (f'{currency}_RSI_{RSI_length}', currency)] <= 30:
            new_positions[currency] = 100
        elif df.loc[current_index, (f'{currency}_RSI_{RSI_length}', currency)] >= 70:
            new_positions[currency] = -100
        else:
            new_positions[currency] = 0
    return new_positions


# code
RSI_length = 14
EMA_length = 21
currencies = ['EURUSD', 'USDJPY', 'EURJPY']  # You can add more currencies here 'EURJPY'

df = load_data(currencies=currencies, timestamp_x='30min')

for currency in currencies:
    df = add_indicators(df, currency=currency, RSI_length=RSI_length, EMA_length=EMA_length)
    df = df.drop(columns=[('Low', currency), ('Open', currency), ('High', currency)])

# Usage example
# Assuming you have your df ready
df = df.dropna()
df = df['2022-06-01':]
result_df = backtest(df,
                     simple_rsi_strategy,
                     starting_capital=10000,
                     provision=0.0001,
                     leverage=30,
                     currencies=['EURUSD', 'USDJPY', 'EURJPY'],
                     RSI_length=RSI_length)

positions_df = pd.DataFrame(columns=['Currency',
                                     'Position',
                                     'Stop_Loss',
                                     'Take_profit',
                                     'Trailing_SL',
                                     'Time_stop',
                                     'Active'])
# 3 trading back testings:
# 1. simple -1 0 1
# 2. different values of positions
# 3. possibility to add stop loss, take profit, trailing stop loss, time stop
