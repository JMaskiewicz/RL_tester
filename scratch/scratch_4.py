# import libraries
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# functions
from data.function.load_data import load_data
from technical_analysys import add_indicators


# code
def backtest_medium(df, strategy, starting_capital=10000, provision=0.0001, leverage=10,
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
        ordered_positions = [new_positions.get(currency, 0) for currency in currencies]
        df.loc[current_index, ('Position_proposition', currencies)] = ordered_positions

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


class SimpleRSIStrategy:
    def __init__(self, RSI_length, stop_loss_percentage=0.01, take_profit_percentage=0.01, trailing_stop_loss_percentage=0.005, time_stop=10, close_on_weekends=True):
        self.RSI_length = RSI_length
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.trailing_stop_loss_percentage = trailing_stop_loss_percentage
        self.open_positions = {}
        self.entry_prices = {}
        self.position_durations = {}
        self.max_gains = {}
        self.time_stop = time_stop
        self.position_log = []
        self.close_on_weekends = close_on_weekends

    def __call__(self, df, current_index, currencies):
        new_positions = {}
        for currency in currencies:
            rsi_value = df.loc[current_index, (f'{currency}_RSI_{self.RSI_length}', currency)]
            current_price = df.loc[current_index, ('Close', currency)]

            # Increase position duration or set it
            self.position_durations[currency] = self.position_durations.get(currency, 0) + 1

            # Check if it's a weekend and close_on_weekends is True
            if self.close_on_weekends and current_index.weekday() >= 5:  # 5 and 6 corresponds to Saturday and Sunday
                if currency in self.open_positions:
                    self.log_position(
                        currency,
                        self.entry_prices[currency]['Date'],
                        current_index,
                        self.entry_prices[currency]['Price'],
                        current_price,
                        self.open_positions[currency],
                        self.open_positions[currency] * (current_price - self.entry_prices[currency]['Price']),
                        "Weekend Stop"
                    )
                    self.cleanup_position(currency)
                continue

            # If position exists for the currency
            if currency in self.open_positions:
                current_gain = self.open_positions[currency] * (current_price - self.entry_prices[currency]['Price'])

                max_gain_for_currency = self.max_gains.get(currency, 0)
                if current_gain > max_gain_for_currency:
                    self.max_gains[currency] = current_gain

                # Check if position duration hit the time_stop
                if self.position_durations[currency] >= self.time_stop:
                    self.log_position(
                        currency,
                        self.entry_prices[currency]['Date'],
                        current_index,
                        self.entry_prices[currency]['Price'],
                        current_price,
                        self.open_positions[currency],
                        current_gain,
                        "Time Stop"
                    )
                    self.cleanup_position(currency)

                # Check for take profit
                elif current_gain >= self.take_profit_percentage:
                    self.log_position(
                        currency,
                        self.entry_prices[currency]['Date'],
                        current_index,
                        self.entry_prices[currency]['Price'],
                        current_price,
                        self.open_positions[currency],
                        current_gain,
                        "Take Profit"
                    )
                    self.cleanup_position(currency)

                # Check for stop loss or trailing stop loss for Long position
                elif self.open_positions[currency] == 100 and (
                        current_price <= self.entry_prices[currency]['Price'] * (1 - self.stop_loss_percentage) or
                        current_price <= self.entry_prices[currency][
                            'Price'] + max_gain_for_currency - self.trailing_stop_loss_percentage
                ):
                    self.log_position(
                        currency,
                        self.entry_prices[currency]['Date'],
                        current_index,
                        self.entry_prices[currency]['Price'],
                        current_price,
                        self.open_positions[currency],
                        current_gain,
                        "Stop Loss" if current_price <= self.entry_prices[currency]['Price'] * (
                                    1 - self.stop_loss_percentage) else "Trailing Stop Loss"
                    )
                    self.cleanup_position(currency)

                # Check for stop loss or trailing stop loss for Short position
                elif self.open_positions[currency] == -100 and (
                        current_price >= self.entry_prices[currency]['Price'] * (1 + self.stop_loss_percentage) or
                        current_price >= self.entry_prices[currency][
                            'Price'] - max_gain_for_currency + self.trailing_stop_loss_percentage
                ):
                    self.log_position(
                        currency,
                        self.entry_prices[currency]['Date'],
                        current_index,
                        self.entry_prices[currency]['Price'],
                        current_price,
                        self.open_positions[currency],
                        current_gain,
                        "Stop Loss" if current_price >= self.entry_prices[currency]['Price'] * (
                                    1 + self.stop_loss_percentage) else "Trailing Stop Loss"
                    )
                    self.cleanup_position(currency)

            # If RSI conditions are met, open new positions (only if there's no open position for that currency)
            else:
                if rsi_value <= 30:
                    new_positions[currency] = 100
                elif rsi_value >= 70:
                    new_positions[currency] = -100
                else:
                    new_positions[currency] = 0

                # Store the entry price, reset the position duration, and store the position
                self.entry_prices[currency] = {"Date": current_index, "Price": current_price}
                self.position_durations[currency] = 0
                self.open_positions[currency] = new_positions[currency]

        return new_positions

    def log_position(self, currency, entry_date, exit_date, entry_price, exit_price, position_size, PnL, reason):
        position_duration = self.position_durations[currency]
        profit_percentage = (PnL / entry_price) * 100 if entry_price != 0 else 0

        # Calculate the difference in time between the exit and entry dates
        duration = exit_date - entry_date

        # Convert the duration to days, hours, and minutes
        days, remainder = divmod(duration.total_seconds(), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes = remainder // 60
        formatted_duration = "{}d: {}h: {}m".format(int(days), int(hours), int(minutes))

        position_record = {
            'Currency': currency,
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Duration (observations)': position_duration,
            'Duration (Days: hours: minutes)': formatted_duration,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Position Size': position_size,
            'PnL': PnL,
            'Profit (%)': profit_percentage,
            'Exit Reason': reason
        }
        self.position_log.append(position_record)

    def cleanup_position(self, currency):
        self.open_positions.pop(currency, None)
        self.position_durations.pop(currency, None)
        self.entry_prices.pop(currency, None)
        self.max_gains.pop(currency, None)

    def get_position_report(self):
        return pd.DataFrame(self.position_log)

    def generate_report(self):
        report = {}
        df_log = pd.DataFrame(self.position_log)

        if not df_log.empty:
            total_trades = len(df_log)
            winning_trades = len(df_log[df_log['PnL'] > 0])
            losing_trades = len(df_log[df_log['PnL'] < 0])

            report['Total Trades'] = total_trades
            report['Winning Trades'] = winning_trades
            report['Losing Trades'] = losing_trades
            report['Win Rate (%)'] = (winning_trades / total_trades) * 100
            report['Average Time of Trade (days)'] = df_log['Duration (Days: hours: minutes)'].str.split('d').str[0].astype(int).mean()
            report['Average Time of Trade (observations)'] = df_log['Duration (observations)'].mean()
            report['Average Gain ($)'] = df_log['PnL'].mean()
            report['Sharpe Ratio'] = df_log['PnL'].mean() / df_log['PnL'].std() if df_log['PnL'].std() != 0 else 0

        return report

# code
RSI_length = 14
EMA_length = 21
currencies = ['EURUSD', 'USDJPY', 'EURJPY']  # You can add more currencies here 'EURJPY'

df = load_data(currencies=currencies, timestamp_x='30min')

for currency in currencies:
    df = add_indicators(df, currency=currency, RSI_length=RSI_length, EMA_length=EMA_length)
    df = df.drop(columns=[('Low', currency), ('Open', currency), ('High', currency)])

rsi_strategy = SimpleRSIStrategy(RSI_length=14)

# Usage example
# Assuming you have your df ready
df = df.dropna()
df = df['2022-06-01':]
result_df = backtest_medium(df,
                     rsi_strategy,
                     starting_capital=10000,
                     provision=0.0001,
                     leverage=30,
                     currencies=['EURUSD', 'USDJPY', 'EURJPY'])

report_df_positions = rsi_strategy.get_position_report()
print(report_df_positions)
report = rsi_strategy.generate_report()
print(report)