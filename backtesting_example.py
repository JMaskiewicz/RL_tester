# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data.function.load_data import load_data
from technical_analysys.add_indicators import add_indicators, compute_volatility
from backtest.backtest_functions.backtest import Strategy


class SimpleRSIStrategy(Strategy):
    def __init__(self, currencies, RSI_length, leverage=1.0, provision=0.0001, starting_capital=10000):
        super().__init__(currencies, leverage, provision, starting_capital)
        self.RSI_length = RSI_length

    def calculate_new_positions(self, df, current_index):
        new_positions = {}
        for currency in self.currencies:
            rsi_value = df.loc[current_index, (f'{currency}_RSI_{self.RSI_length}', currency)]
            if rsi_value <= 30:
                new_positions[currency] = (30 - round(rsi_value)) * 100
            elif rsi_value >= 70:
                new_positions[currency] = - round(rsi_value - 70) * 100
            else:
                new_positions[currency] = 0  # No open position

        return new_positions

# Usage Example
RSI_length = 21
currencies = ['EURUSD', 'EURJPY', 'USDJPY']
df = load_data(currencies=currencies, timestamp_x='4h')

for currency in currencies:
    df = add_indicators(df, currency=currency, RSI=RSI_length)

df = df.dropna()
df = df['2022-01-01':]

rsi_strategy = SimpleRSIStrategy(currencies, RSI_length=RSI_length, starting_capital=10000)
df = rsi_strategy.backtest(df)

report_df_positions = rsi_strategy.generate_report(df)
report = rsi_strategy.generate_extended_report()
rsi_strategy.display_summary_as_table(df, extended_report=True)

# Plotting
from backtest.plots.plot import plot_financial_data

plot_financial_data(df, rsi_strategy, currencies, volatility='garman_klass_volatility', n=200)

