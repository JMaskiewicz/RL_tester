# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Strategy:
    def __init__(self, df, tradable_markets, leverage=1.0, provision=0.0001, starting_capital=10000):
        self.trade_counter = 0
        self.leverage = leverage
        self.provision = provision
        self.starting_capital = starting_capital
        self.tradable_markets = tradable_markets
        self.df = df
        self.trade_history = pd.DataFrame(columns=['Trade_ID',
                                                   'Parent_ID',
                                                   'Currency',
                                                   'Position_Size',
                                                   'Unit_Size',
                                                   'Open_Price',
                                                   'Close_Price',
                                                   'Open_Date',
                                                   'Close_Date',
                                                   'Take_Profit',
                                                   'Stop_Loss',
                                                   'Trailing_stop_loss',
                                                   'Time_stop_loss',
                                                   'Provisions',
                                                   'PnL',
                                                   'Exit Reason'])

    def calculate_new_positions(self, current_index):
        raise NotImplementedError("Subclasses should implement this method")

    def backtest(self):
        current_index = self.df.index[0]
        self.df['Capital'] = 0
        self.df.loc[current_index, 'Capital'] = self.starting_capital
        self.df['Available_Capital'] = 0
        self.df.loc[current_index, 'Available_Capital'] = self.starting_capital
        self.df['Margin'] = 1
        self.df['Provisions'] = 0

        for market in self.tradable_markets:
            self.df[('Position_proposition', currency)] = 0
            self.df[('Capital_in', currency)] = 0
            self.df[('Average_Open_Position', currency)] = 0
            self.df[('Unit_size'), currency] = 0
            self.df[('PnL', currency)] = 0

        for i in tqdm(range(1, len(df))):
            previous_index = current_index
            current_index = df.index[i]

            capital_in_previous = df.loc[previous_index, ('Capital_in', self.currencies)]
            df.loc[current_index, 'Available_Capital'] = df.loc[previous_index, 'Capital'][0] - abs(
                capital_in_previous).sum()

            new_positions = self.calculate_new_positions(df, current_index)

            ordered_positions = [new_positions.get(currency, 0) for currency in self.currencies]
            df.loc[current_index, ('Position_proposition', self.currencies)] = ordered_positions

            new_positions_series = pd.Series(new_positions).reindex(capital_in_previous.index.get_level_values(1))
            required_capital_series = abs(new_positions_series + capital_in_previous)
            total_required_capital = required_capital_series.sum()

            new_provisions_series = pd.Series(new_positions).reindex(capital_in_previous.index.get_level_values(1))
            provisions_series = abs(new_provisions_series) * self.provision
            total_provisions = provisions_series.sum()

            current_close_prices = df.loc[current_index, ('Close', self.currencies)]
            previous_close_prices = df.loc[previous_index, ('Close', self.currencies)]
            current_units = df.loc[previous_index, ('Unit_size', self.currencies)]

            average_open_position = df.loc[previous_index, ('Average_Open_Position', self.currencies)]
            df.loc[current_index, ('Average_Open_Position', self.currencies)] = average_open_position

            current_returns = (current_close_prices.reset_index(drop=True) - previous_close_prices.reset_index(drop=True))
            current_capital_in = df.loc[previous_index, ('Capital_in', self.currencies)].reset_index(drop=True)
            current_capital_in.index = current_units.index
            current_returns.index = current_units.index

            PnL_series = (current_returns * current_units * self.leverage).fillna(0) * np.sign(current_capital_in)
            total_PnL = PnL_series.loc['Unit_size'].sum()

            df.loc[current_index, ('PnL', self.currencies)] = PnL_series.loc['Unit_size'].values
            df.loc[current_index, ('Capital_in', self.currencies)] = df.loc[previous_index, ('Capital_in', self.currencies)]
            df.loc[current_index, ('Unit_size', self.currencies)] = df.loc[previous_index, ('Unit_size', self.currencies)]

            if total_required_capital <= df.loc[previous_index, 'Capital'][0]:
                df.loc[current_index, 'Provisions'] = total_provisions

                # editing position log
                for currency in self.currencies:
                    proposed_position = df.loc[current_index, ('Position_proposition', currency)]
                    current_position = df.loc[previous_index, ('Capital_in', currency)]

                    # If there's a change in position
                    if proposed_position != 0:
                        if np.sign(proposed_position) == np.sign(current_position) or current_position == 0:
                            self.record_trade(currency, proposed_position, current_index, df)
                            df.loc[current_index, ('Average_Open_Position', currency)] = (df.loc[previous_index, ('Average_Open_Position', currency)] * abs(current_position) + abs(proposed_position) * current_close_prices['Close', currency]) / (abs(current_position) + abs(proposed_position))
                            df.loc[current_index, ('Unit_size', currency)] = current_units['Unit_size', currency] + abs(proposed_position) / current_close_prices['Close', currency]
                        elif np.sign(proposed_position) != np.sign(current_position) and abs(proposed_position) > abs(
                                current_position):
                            self.close_trade(currency, current_position, current_index, df)
                            self.record_trade(currency, proposed_position + current_position, current_index, df)
                            df.loc[current_index, ('Average_Open_Position', currency)] = current_close_prices['Close', currency]
                            df.loc[current_index, ('Unit_size', currency)] = abs(proposed_position + current_position) / current_close_prices['Close', currency]
                        else:
                            self.close_trade(currency, proposed_position, current_index, df)

                df.loc[current_index, ('Capital_in', self.currencies)] = new_positions_series + capital_in_previous


            else:
                net_change_positions = abs(capital_in_previous + new_positions_series) - abs(capital_in_previous)
                net_change_positions_flat = net_change_positions.reset_index(level=0, drop=True)
                net_change_positions_flat = net_change_positions_flat.reindex(new_positions_series.index)

                closing_positions = new_positions_series.where(net_change_positions_flat < 0, 0)
                smaller_abs_values = pd.Series(
                    np.minimum(np.abs(closing_positions.values), np.abs(capital_in_previous.values)),
                    index=closing_positions.index) * np.sign(new_positions_series)

                total_closing_capital = abs(smaller_abs_values).sum()

                if total_closing_capital > 0:
                    total_provisions = abs(closing_positions).sum() * self.provision
                    df.loc[current_index, 'Provisions'] = total_provisions

                    for currency in self.currencies:
                        closing = closing_positions[currency]
                        if closing != 0:
                            self.close_trade(currency, closing, current_index, df)
                else:
                    total_provisions = 0

                df.loc[current_index, ('Capital_in', self.currencies)] = capital_in_previous + smaller_abs_values

            df.loc[current_index, 'Capital'] = df.loc[previous_index, 'Capital'][0] - total_provisions + total_PnL

        for currency in self.currencies:
            end_closing = df.loc[current_index, ('Capital_in', currency)]
            if end_closing != 0:
                self.close_trade(currency, end_closing, current_index, df)

        return df

    def record_trade(self, currency, position, current_index, df, parent_id=None):
        open_price = df.loc[current_index, ('Close', currency)]
        open_date = current_index

        self.trade_counter += 1
        trade_id = self.trade_counter

        if parent_id is None:
            parent_id = trade_id

        trade_info = {
            'Trade_ID': trade_id,
            'Parent_ID': parent_id,  # Add the unique trade ID here
            'Currency': currency,
            'Position_Size': position,
            'Unit_Size': abs(position) / open_price,
            'Open_Price': open_price,
            'Close_Price': None,  # to be filled when the position is closed
            'Open_Date': open_date,
            'Close_Date': None,  # to be filled when the position is closed
            'PnL': None,  # to be filled when the position is closed
            'Exit Reason': 'New Position',  # Exit Reason for the trade (open/close/etc.)
            'Status': 'open'  # we need to track the status of each trade
        }

        # Append the trade to the history DataFrame
        self.trade_history = pd.concat([self.trade_history, pd.DataFrame([trade_info])], ignore_index=True)


    def close_trade(self, currency, proposed_position, current_index, df):
        # Find all open trades for the given currency
        open_trades = self.trade_history[
            (self.trade_history['Currency'] == currency) &
            (self.trade_history['Status'] == 'open')]

        open_trades = open_trades.sort_values(by=['Open_Date'], ascending=True)

        if not open_trades.empty:
            remaining_position_to_close = proposed_position

            open_trades_sum = open_trades['Position_Size'].sum()
            current_capital_in = df.loc[current_index, ('Capital_in', currency)]
            current_unit_size = df.loc[current_index, ('Unit_size', currency)]

            if current_capital_in != open_trades_sum:
                print(f"Current capital: {current_capital_in}, Open trades sum: {open_trades_sum}")
                print('wrong')

            # Update each open trade
            for index, trade in open_trades.iterrows():
                current_position_size = trade['Position_Size']

                # If the remaining position to close is zero, we stop processing
                if remaining_position_to_close == 0:
                    break

                # Determine the size of the position we are going to close
                position_to_close = min(abs(current_position_size), abs(remaining_position_to_close)) * np.sign(current_position_size)

                # Adjust the remaining position size to close
                remaining_position_to_close += position_to_close

                # Update the record in the trade history
                close_price = df.loc[current_index, ('Close', currency)]
                pnl = position_to_close / trade['Open_Price'] * (close_price - trade['Open_Price']) * self.leverage

                average_open_position = df.loc[current_index, ('Average_Open_Position', currency)]

                if position_to_close == current_position_size:
                    # If we're closing the entire position, update the status to 'closed'
                    self.trade_history.at[index, 'Status'] = 'closed'
                    self.trade_history.at[index, 'Close_Price'] = close_price
                    self.trade_history.at[index, 'Close_Date'] = current_index
                    self.trade_history.at[index, 'PnL'] = pnl
                    self.trade_history.at[index, 'Exit Reason'] = 'Close Position'

                    # add editing average open position
                    current_capital_in -= position_to_close
                    current_unit_size -= abs(position_to_close) / trade['Open_Price']
                    if current_capital_in == 0:
                        df.loc[current_index, ('Average_Open_Position', currency)] = 0
                        df.loc[current_index, ('Unit_size', currency)] = 0
                    else:
                        df.loc[current_index, ('Average_Open_Position', currency)] = (average_open_position * (abs(current_capital_in) + abs(position_to_close)) - abs(position_to_close) *self.trade_history.at[index, 'Open_Price']) / abs(current_capital_in)
                        df.loc[current_index, ('Unit_size', currency)] = current_unit_size

                else:
                    # We're partially closing the position, so we need to update the position size
                    # Create a new record for the remaining position
                    remaining_position = current_position_size - position_to_close
                    new_trade_record = trade.copy()
                    new_trade_record['Position_Size'] = remaining_position
                    new_trade_record['Unit_Size'] = abs(remaining_position) / trade['Open_Price']
                    new_trade_record['Trade_ID'] = self.trade_counter + 1  # unique new ID
                    self.trade_counter += 1

                    # Update the original record to reflect the partial close
                    self.trade_history.at[index, 'Status'] = 'closed'
                    self.trade_history.at[index, 'Position_Size'] = position_to_close
                    self.trade_history.at[index, 'Unit_Size'] = abs(position_to_close) / trade['Open_Price']
                    self.trade_history.at[index, 'Close_Price'] = close_price
                    self.trade_history.at[index, 'Close_Date'] = current_index
                    self.trade_history.at[index, 'PnL'] = pnl
                    self.trade_history.at[index, 'Exit Reason'] = 'Partial Close'

                    # Add the new record for the remaining open position to the history
                    self.trade_history = pd.concat([self.trade_history, pd.DataFrame([new_trade_record])],ignore_index=True)

                    current_capital_in -= position_to_close
                    current_unit_size -= abs(position_to_close) / trade['Open_Price']

                    if current_capital_in == 0:
                        print('dupa')

                    # add editing average open position
                    df.loc[current_index, ('Average_Open_Position', currency)] = (average_open_position * (abs(current_capital_in) + abs(position_to_close)) - abs(position_to_close) *self.trade_history.at[index, 'Open_Price']) / abs(current_capital_in)
                    df.loc[current_index, ('Unit_size', currency)] = current_unit_size

    def calculate_sortino_ratio(self, df, target_return=0):
        # Total returns are the increases in capital over time
        total_returns = df['Capital'].pct_change().fillna(0)

        # Downside deviation
        negative_returns = total_returns[total_returns < target_return]
        downside_deviation = negative_returns.std()

        # Calculate the mean of returns
        mean_return = total_returns.mean()

        # Sortino Ratio calculation
        sortino_ratio = (mean_return - target_return) / downside_deviation if downside_deviation != 0 else np.nan

        return sortino_ratio

    def calculate_max_drawdown(self, df):
        capital = df['Capital']
        peak = capital.expanding(min_periods=1).max()
        drawdown = (capital - peak) / peak

        return drawdown.min()

    def generate_report(self, df):
        # Set Pandas options and initialize the report
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        report = {}
        df_log = pd.DataFrame(self.trade_history)

        if not df_log.empty:
            total_trades = len(df_log)
            winning_trades = len(df_log[df_log['PnL'] > 0])
            losing_trades = len(df_log[df_log['PnL'] < 0])

            total_pnl = df['PnL'].sum(axis=1)

            report['Start Date'] = df.index[0]
            report['End Date'] = df.index[-1]
            report['Total PnL'] = df['PnL'].sum().sum()
            report['Total Return'] = df['Capital'].iloc[-1] / df['Capital'].iloc[0] - 1
            report['Total Provisions'] = df['Provisions'].sum().sum()
            report['Sortino Ratio'] = self.calculate_sortino_ratio(df)
            report['Sharpe Ratio'] = total_pnl.mean() / total_pnl.std() if total_pnl.std() != 0 else 0
            report['Max Drawdown'] = self.calculate_max_drawdown(df)
            report['Max Drawdown Duration'] = df['Capital'].idxmin() - df['Capital'].idxmax()
            report['Total Trades'] = total_trades
            report['Winning Trades'] = winning_trades
            report['Losing Trades'] = losing_trades
            report['Win Rate (%)'] = (winning_trades / total_trades) * 100
            report['Average Gain ($)'] = df_log['PnL'].mean()

            report['Average Gain (%)'] = self.compute_average_gain_percentage(df_log)
            report['Average Loss ($)'] = df_log[df_log['PnL'] < 0]['PnL'].mean()
            report['Average Loss (%)'] = df_log[df_log['PnL'] < 0]['PnL'].mean() / df_log[df_log['PnL'] < 0]['Open_Price'].mean() * 100
            report['Average Risk Reward'] = abs(report['Average Gain ($)'] / report['Average Loss ($)']) if report['Average Loss ($)'] != 0 else 0
            report['Average Risk Reward (%)'] = abs(report['Average Gain (%)'] / report['Average Loss (%)']) if report['Average Loss (%)'] != 0 else 0
            report['Average Risk Reward Ratio'] = abs(report['Average Gain (%)'] / report['Average Loss (%)']) if report['Average Loss (%)'] != 0 else 0
            report['Average Holding Time'] = pd.to_datetime(df_log['Close_Date']).mean() - pd.to_datetime(df_log['Open_Date']).mean()

            # Breakdown by 'Exit Reason'
            exit_reason_counts = df_log['Exit Reason'].value_counts()
            exit_reason_percentage = exit_reason_counts / total_trades * 100
            exit_reason_report = {}
            for reason, count in exit_reason_counts.items():
                percentage = exit_reason_percentage[reason]
                exit_reason_report[f"{reason}"] = f"{count} times ({percentage:.2f}%)"
            report['Exit Reason'] = exit_reason_report

        return report

    def generate_extended_report(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        full_report = {}
        df_log = pd.DataFrame(self.trade_history)

        for currency in df_log['Currency'].unique():
            df_currency_log = df_log[df_log['Currency'] == currency]

            if not df_currency_log.empty:
                report = {}
                total_trades = len(df_currency_log)
                winning_trades = len(df_currency_log[df_currency_log['PnL'] > 0])
                losing_trades = len(df_currency_log[df_currency_log['PnL'] < 0])

                report['Total Trades'] = total_trades
                report['Winning Trades'] = winning_trades
                report['Losing Trades'] = losing_trades
                report['Win Rate (%)'] = (winning_trades / total_trades) * 100
                report['Average Gain ($)'] = df_currency_log['PnL'].mean()

                # Calculate Total PnL after provisions for the specific currency
                total_pnl_before_provision = df_currency_log['PnL'].sum()

                # Ensure the provision column exists, if not consider it as 0.
                if 'Provision' in df_currency_log.columns:
                    total_provision_for_currency = df_currency_log['Provision'].sum()
                else:
                    total_provision_for_currency = 0

                total_pnl_after_provision = total_pnl_before_provision - total_provision_for_currency

                report['Total PnL before Provisions'] = total_pnl_before_provision
                report['Total Provisions for ' + currency] = total_provision_for_currency
                report['Total PnL after Provisions'] = total_pnl_after_provision

                # Calculate the average gain in percentage
                report['Average Gain (%)'] = self.compute_average_gain_percentage(df_currency_log)

                # Compute the Sharpe Ratio using the Average Gain in %
                mean_percentage_return = report['Average Gain (%)']
                std_dev_percentage_return = (df_currency_log['PnL'] / df_currency_log['Open_Price']).std() * 100
                report['Sharpe Ratio'] = mean_percentage_return / std_dev_percentage_return if std_dev_percentage_return != 0 else 0

                # Breakdown by 'Exit Reason'
                exit_reason_counts = df_currency_log['Exit Reason'].value_counts()
                exit_reason_percentage = exit_reason_counts / total_trades * 100
                exit_reason_report = {}
                for reason, count in exit_reason_counts.items():
                    percentage = exit_reason_percentage[reason]
                    exit_reason_report[f"{reason}"] = f"{count} times ({percentage:.2f}%)"
                report['Exit Reason'] = exit_reason_report

                full_report[currency] = report

        return full_report

    def compute_average_gain_percentage(self, df_currency_log):
        total_gain_percentage = (df_currency_log['PnL'] / abs(df_currency_log['Open_Price'])).mean()
        return total_gain_percentage * 100

    def display_summary_as_table(self, df, extended_report=False):
        # The standard report is always generated first.
        standard_report = self.generate_report(df)

        # Prepare the DataFrame for the standard report.
        standard_rows_list = [{'Metric': key, 'Value': value} for key, value in standard_report.items()]
        df_summary = pd.DataFrame(standard_rows_list)  # This is your standard report in DataFrame form.

        # If the extended report flag is set, we generate the extended report and append its information.
        if extended_report:
            extended_summary = self.generate_extended_report()

            for currency, currency_report in extended_summary.items():
                # This is a header row for the currency section.
                df_summary = pd.concat([df_summary, pd.DataFrame([{'Metric': f'--- {currency} ---', 'Value': ''}])], ignore_index=True)

                # These are the detailed rows for the currency section.
                currency_rows_list = [{'Metric': key, 'Value': value} for key, value in currency_report.items()]
                df_summary = pd.concat([df_summary, pd.DataFrame(currency_rows_list)], ignore_index=True)

        # Display the final table, which includes the standard report and, if applicable, the extended reports.
        print(df_summary.to_string(index=False))