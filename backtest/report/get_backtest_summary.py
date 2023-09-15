# import libraries
import numpy as np

def get_backtest_summary(df, currencies):
    summary = {}

    # PnL
    summary['Total PnL'] = df['Capital'].iloc[-1] - df['Capital'].iloc[0]

    # Start and End Date
    summary['Start Date'] = df.index[0]
    summary['End Date'] = df.index[-1]

    # Start and End Capital
    summary['Start Capital'] = df['Capital'].iloc[0]
    summary['End Capital'] = df['Capital'].iloc[-1]

    # Profit in %
    summary['Profit %'] = ((summary['End Capital'] - summary['Start Capital']) / summary['Start Capital']) * 100

    # Profit for Each Currency
    summary['Profit per Currency'] = {currency: df[('PnL', currency)].sum() for currency in currencies}

    # Total Provisions
    summary['Total Provisions'] = - df['Provisions'].sum()

    # Sortino Ratio (simplified, assumes risk-free rate is 0)
    daily_returns = df['Capital'].pct_change().dropna()
    negative_std = daily_returns[daily_returns < 0].std()
    sortino_ratio = daily_returns.mean() / negative_std
    summary['Sortino Ratio'] = sortino_ratio

    # Max Drawdown and Drawdown Duration
    rolling_max = df['Capital'].cummax()
    drawdown = df['Capital'] / rolling_max - 1.0
    max_drawdown = drawdown.min()
    end_dd = drawdown.idxmin()
    start_dd = rolling_max[:end_dd].idxmax()
    duration = (end_dd - start_dd).days
    summary['Max Drawdown'] = max_drawdown
    summary['Drawdown Duration'] = duration

    # Average Drawdown
    summary['Average Drawdown'] = drawdown.mean()

    # Average Drawdown Duration
    durations = []
    in_drawdown = False
    start_date = None
    for date, value in drawdown.items():
        if value < 0:
            if not in_drawdown:
                start_date = date
                in_drawdown = True
        elif in_drawdown:
            end_date = date
            durations.append((end_date - start_date).days)
            in_drawdown = False
    summary['Average Drawdown Duration'] = sum(durations) / len(durations) if durations else 0

    # Calmar Ratio (annualized return / max drawdown)
    trading_days_per_year = 252  # typical number for stocks, adjust if needed for forex
    annualized_return = ((summary['End Capital'] / summary['Start Capital']) ** (trading_days_per_year / len(df))) - 1
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    summary['Calmar Ratio'] = calmar_ratio

    return summary