# import libraries
import numpy as np
import pandas as pd

from backtest.report.get_backtest_summary import get_backtest_summary

def display_summary_as_table(backtest_result, strategy, currencies, extended_report=False):
    summary = get_backtest_summary(backtest_result, currencies)
    trade_report = strategy.generate_report()
    extended_report = strategy.generate_extended_report() if extended_report else {}

    df_summary = pd.DataFrame(columns=['Metric', 'Value'])

    # First, add the main summary metrics
    for key, value in summary.items():
        if key == 'Profit per Currency':
            for currency, profit in value.items():
                df_summary.loc[len(df_summary)] = {'Metric': f'Profit ({currency})', 'Value': profit}
        else:
            df_summary.loc[len(df_summary)] = {'Metric': key, 'Value': value}

    # Add a separator for Trade Statistics
    df_summary.loc[len(df_summary)] = {'Metric': '# TRADES STATISTICS', 'Value': ''}

    # Then, add the trade related metrics from the report
    for key, value in trade_report.items():
        if key == 'Exit Reason Breakdown':
            for reason, stats in value.items():
                df_summary.loc[len(df_summary)] = {'Metric': reason, 'Value': stats}
        else:
            df_summary.loc[len(df_summary)] = {'Metric': key, 'Value': value}

    if extended_report:
        df_summary.loc[len(df_summary)] = {'Metric': '# EXTENDED REPORT', 'Value': ''}

        for currency, currency_report in extended_report.items():
            # Add a separator for Trade Statistics by Currency
            df_summary.loc[len(df_summary)] = {'Metric': f'# TRADES STATISTICS CURRENCY ({currency})', 'Value': ''}
            for key, value in currency_report.items():
                if key == 'Exit Reason Breakdown':
                    for reason, stats in value.items():
                        df_summary.loc[len(df_summary)] = {'Metric': reason, 'Value': stats}
                else:
                    df_summary.loc[len(df_summary)] = {'Metric': key, 'Value': value}
    return df_summary