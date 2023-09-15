# import libraries
import numpy as np
import pandas as pd

from backtest.backtest_functions.backtest import backtest
from backtest.report.get_backtest_summary import get_backtest_summary
from backtest.display_summary_as_table import display_summary_as_table
from Strategy.Strategy import Strategy
from data.function.load_data import load_data
from technical_analysys.add_indicators import add_indicators, compute_volatility


class SimpleRSIStrategy(Strategy):
    def __init__(self, RSI_length, stop_loss_percentage=0.01):
        super().__init__()  # Initialize attributes from the BaseStrategy class
        self.RSI_length = RSI_length
        self.stop_loss_percentage = stop_loss_percentage

    def __call__(self, df, current_index, currencies):
        new_positions = {}
        for currency in currencies:
            rsi_value = df.loc[current_index, (f'{currency}_RSI_{self.RSI_length}', currency)]
            current_price = df.loc[current_index, ('Close', currency)]

            if currency in self.open_positions:
                stop_loss_hit = False

                # Check for Stop Loss
                if self.open_positions[currency] > 0:  # for a long position
                    if current_price <= self.entry_prices[currency]['Price'] * (1 - self.stop_loss_percentage):
                        stop_loss_hit = True
                elif self.open_positions[currency] < 0:  # for a short position
                    if current_price >= self.entry_prices[currency]['Price'] * (1 + self.stop_loss_percentage):
                        stop_loss_hit = True

                # Check for RSI closure or Stop Loss hit
                if (rsi_value > 30 and self.open_positions[currency] > 0) or \
                        (rsi_value < 70 and self.open_positions[currency] < 0) or \
                        stop_loss_hit:

                    reason = "Stop Loss Hit" if stop_loss_hit else "Position Closed"
                    current_gain = self.open_positions[currency] * (
                            current_price - self.entry_prices[currency]['Price'])
                    self.log_position(
                        currency,
                        self.entry_prices[currency]['Date'],
                        current_index,
                        self.entry_prices[currency]['Price'],
                        current_price,
                        self.open_positions[currency],
                        current_gain,
                        reason
                    )
                    self.cleanup_position(currency)
                    new_positions[currency] = 0  # No open position after closing
                else:
                    new_positions[currency] = self.open_positions[currency]  # Position remains unchanged
            else:
                if rsi_value <= 30:
                    new_positions[currency] = (30 - round(rsi_value)) * 100
                elif rsi_value >= 70:
                    new_positions[currency] = - round((rsi_value) - 70) * 100
                else:
                    new_positions[currency] = 0  # No open position
                if new_positions[currency] != 0:
                    self.entry_prices[currency] = {"Date": current_index, "Price": current_price}
                    self.open_positions[currency] = new_positions[currency]

        return new_positions


# code
RSI_length = 21
currencies = ['EURUSD', 'USDJPY', 'EURJPY']  # You can add more currencies here 'EURJPY'

df = load_data(currencies=currencies, timestamp_x='4h')

for currency in currencies:
    df = add_indicators(df, currency=currency, RSI=RSI_length)
    # df = df.drop(columns=[('Low', currency), ('Open', currency), ('High', currency)])

rsi_strategy = SimpleRSIStrategy(RSI_length=RSI_length, stop_loss_percentage=0.05)

# Usage example
# Assuming you have your df ready
df = df.dropna()
df = df['2022-06-01':]
backtest_result = backtest(df,
                           rsi_strategy,
                           starting_capital=100000,
                           provision=0.0001,
                           leverage=10,
                           currencies=['EURUSD', 'USDJPY', 'EURJPY'])

report_df_positions = rsi_strategy.get_position_report()
report = rsi_strategy.generate_report()
summary = get_backtest_summary(backtest_result, currencies)

df_summary = display_summary_as_table(backtest_result, rsi_strategy, currencies, extended_report=True)
print(summary)
print(report)
print(report_df_positions)
pd.set_option('display.max_rows', None)
print(df_summary)
pd.reset_option('display.max_rows')

# PLOTS
from backtest.plots.plot import plot_financial_data

plot_financial_data(df, rsi_strategy, currencies)
