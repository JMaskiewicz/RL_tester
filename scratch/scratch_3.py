import random
import pandas as pd
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# functions
from data.function.load_data import load_data
from technical_analysys import add_indicators


def backtest_easy(df, strategy, starting_capital=10000, provision=0.0001, leverage=10, currencies=['EURUSD', 'USDJPY', 'EURJPY'], **kwargs):
    num_currencies = len(currencies)
    capital_per_currency = starting_capital / num_currencies

    current_index = df.index[0]
    df['Capital'] = starting_capital
    df['Position'] = 0
    df['Provisions'] = 0

    for currency in currencies:
        df[('PnL', currency)] = 0

    for i in tqdm(range(1, len(df))):
        previous_index = current_index
        current_index = df.index[i]

        new_positions = strategy(df, current_index, currencies, **kwargs)
        positions_series = pd.Series(new_positions, index=currencies) * capital_per_currency

        for currency in currencies:
            df.loc[current_index, ('Position', currency)] = positions_series[currency]

        provisions_series = abs(positions_series) * provision
        total_provisions = provisions_series.sum()

        current_close_prices = df.loc[current_index, ('Close', currencies)]
        previous_close_prices = df.loc[previous_index, ('Close', currencies)]
        current_returns = (current_close_prices / previous_close_prices - 1)
        PnL_series = current_returns * positions_series * leverage
        total_PnL = PnL_series.sum()

        df.loc[current_index, ('PnL', currencies)] = PnL_series.values
        df.loc[current_index, ('Capital_in', currencies)] = positions_series.values
        df.loc[current_index, 'Provisions'] = total_provisions
        df.loc[current_index, 'Capital'] = df.loc[previous_index, 'Capital'] - total_provisions + total_PnL

    return df

class RandomStickStrategy:
    def __init__(self, duration=10):
        self.duration = duration
        self.counter = {}
        self.current_position = {}

    def initialize_currency(self, currency):
        self.counter[currency] = self.duration
        self.current_position[currency] = 0

    def __call__(self, df, current_index, currencies, **kwargs):
        positions = []
        for currency in currencies:
            if currency not in self.counter:
                self.initialize_currency(currency)

            if self.counter[currency] == self.duration:
                self.current_position[currency] = random.choice([-1, 0, 1])
                self.counter[currency] = 0

            positions.append(self.current_position[currency])
            self.counter[currency] += 1

        return positions

def plot_backtest_results(df):
    plt.figure(figsize=(15, 8))
    df['Capital'].plot()
    plt.title('Backtest Results')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.grid()
    plt.show()

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

# Run the backtest
strategy_instance = RandomStickStrategy()
results = backtest_easy(df, strategy_instance, starting_capital=10000, currencies=['EURUSD', 'USDJPY', 'EURJPY'])

# Visualize the results
plot_backtest_results(results)