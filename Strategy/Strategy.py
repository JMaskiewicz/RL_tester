# import libraries
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
from tqdm import tqdm


class Strategy:
    def __init__(self, leverage=1.0, provision=0.0001):
        self.leverage = leverage
        self.provision = provision
        self.open_positions = {}
        self.entry_prices = {}
        self.position_log = []

    def log_position(self, currency, entry_date, exit_date, entry_price, exit_price, position_size, PnL, reason):
        provision_cost = abs(position_size) * self.provision
        PnL -= 2 * provision_cost  # Provision for both opening and closing the position
        profit_percentage = (PnL / entry_price) * 100 if entry_price != 0 else 0
        duration = exit_date - entry_date
        days, remainder = divmod(duration.total_seconds(), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes = remainder // 60
        formatted_duration = "{}d: {}h: {}m".format(int(days), int(hours), int(minutes))
        position_record = {
            'Currency': currency,
            'Entry Date': entry_date,
            'Exit Date': exit_date,
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
        self.entry_prices.pop(currency, None)

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
            report['Average Gain ($)'] = df_log['PnL'].mean()
            report['Sharpe Ratio'] = df_log['PnL'].mean() / df_log['PnL'].std() if df_log['PnL'].std() != 0 else 0

            # Breakdown by 'Exit Reason'
            exit_reason_counts = df_log['Exit Reason'].value_counts()
            exit_reason_percentage = exit_reason_counts / total_trades * 100
            exit_reason_report = {}
            for reason, count in exit_reason_counts.items():
                percentage = exit_reason_percentage[reason]
                exit_reason_report[f"{reason}"] = f"{count} times ({percentage:.2f}%)"
            report['Exit Reason Breakdown'] = exit_reason_report

        return report

    def compute_average_gain_percentage(self, df_currency_log):
        total_gain_percentage = (df_currency_log['PnL'] / abs(df_currency_log['Entry Price'])).mean()
        return total_gain_percentage * 100

    def generate_extended_report(self):
        full_report = {}
        df_log = pd.DataFrame(self.position_log)

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
                report['Average Time of Trade (days)'] = \
                df_currency_log['Duration (Days: hours: minutes)'].str.split('d').str[0].astype(int).mean()
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
                std_dev_percentage_return = (df_currency_log['PnL'] / df_currency_log['Entry Price']).std() * 100
                report[
                    'Sharpe Ratio'] = mean_percentage_return / std_dev_percentage_return if std_dev_percentage_return != 0 else 0

                # Breakdown by 'Exit Reason'
                exit_reason_counts = df_currency_log['Exit Reason'].value_counts()
                exit_reason_percentage = exit_reason_counts / total_trades * 100
                exit_reason_report = {}
                for reason, count in exit_reason_counts.items():
                    percentage = exit_reason_percentage[reason]
                    exit_reason_report[f"{reason}"] = f"{count} times ({percentage:.2f}%)"
                report['Exit Reason Breakdown'] = exit_reason_report

                full_report[currency] = report

        return full_report