import numpy as np
from tqdm import tqdm
import pandas as pd

from technical_analysys.indicators import rsi, simple_moving_average, average_true_range, macd, stochastic_oscillator, parabolic_sar
from technical_analysys.volatility_functions import close_to_close_volatility, parkinson_volatility, garman_klass_volatility, rogers_satchell_volatility


def add_returns(df, return_indicators):
    """
    Calculates and adds simple returns to the DataFrame for multiple market identifiers and price types.

    Parameters:
    - df: DataFrame with multi-level columns (price type, market identifier).
    - return_indicators: List of dictionaries with 'price_type' and 'mkf' keys.
    """
    for indicator in return_indicators:
        price_type = indicator['price_type']
        mkf = indicator['mkf']

        # Ensure the market and price type are in the DataFrame
        if (price_type, mkf) in df.columns:
            # Calculate simple returns
            df[('Returns_' + price_type, mkf)] = df[(price_type, mkf)].pct_change(1, fill_method=None)  # Assuming n=1 for simplicity
        else:
            print(f"{price_type} data for market {mkf} not found in DataFrame")
    return df


def add_log_returns(df, return_indicators):
    """
    Calculates and adds log returns to the DataFrame for multiple market identifiers and price types.

    Parameters:
    - df: DataFrame with multi-level columns (price type, market identifier).
    - return_indicators: List of dictionaries with 'price_type' and 'mkf' keys.
    """
    for indicator in return_indicators:
        price_type = indicator['price_type']
        mkf = indicator['mkf']

        # Ensure the market and price type are in the DataFrame
        if (price_type, mkf) in df.columns:
            # Calculate log returns
            df[('Log_Returns_' + price_type, mkf)] = np.log(
                df[(price_type, mkf)] / df[(price_type, mkf)].shift(1))  # Assuming n=1 for simplicity
        else:
            print(f"{price_type} data for market {mkf} not found in DataFrame")
    return df

def add_indicators(df, indicators):
    for indicator in tqdm(indicators):
        mkf = indicator["mkf"]
        length = indicator.get("length", 14)  # Default length
        if mkf in df.columns.get_level_values(1):
            if indicator["indicator"].startswith("RSI"):
                df[('RSI_' + str(length), mkf)] = rsi(df, mkf, length)
            elif indicator["indicator"].startswith("SMA"):
                df[('SMA_' + str(length), mkf)] = simple_moving_average(df, mkf, length)
            elif indicator["indicator"].startswith("ATR"):
                df[('ATR_' + str(length), mkf)] = average_true_range(df, mkf, length)
            elif indicator["indicator"].startswith("MACD"):
                macd_line, signal_line = macd(df, mkf)
                df[('MACD_Line', mkf)] = macd_line
                df[('Signal_Line', mkf)] = signal_line
            elif indicator["indicator"].startswith("Stochastic"):
                k_percent, d_percent = stochastic_oscillator(df, mkf)
                df[('K%', mkf)] = k_percent
                df[('D%', mkf)] = d_percent
            elif indicator["indicator"].startswith("ParabolicSAR"):
                df[('Parabolic_SAR', mkf)] = parabolic_sar(df, mkf)
        else:
            print(f"Market {mkf} not found in DataFrame")
    return df


def compute_volatility(df, currency, method_func='close_to_close_volatility', n=50):
    """
    Compute rolling volatility for a given currency using the provided method function.

    Parameters:
        - df: DataFrame with the data.
        - currency: The currency for which volatility needs to be computed.
        - method_func_str: The string name of the function to compute volatility.
        - n: Rolling window size.

    Returns:
        - Pandas Series with computed volatility.
    """

    method_func_dict = {
        'close_to_close_volatility': close_to_close_volatility,
        'parkinson_volatility': parkinson_volatility,
        'garman_klass_volatility': garman_klass_volatility,
        'rogers_satchell_volatility': rogers_satchell_volatility
    }

    if method_func not in method_func_dict:
        raise ValueError(f"Unknown method function string '{method_func}'")

    return method_func_dict[method_func](df, currency, n)


def add_time_sine_cosine(df, timestamp):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)

    # Determine the number of seconds in one 'timestamp' period
    if timestamp.endswith('H'):  # Hour
        period_seconds = 3600
    elif timestamp.endswith('D'):  # Day
        period_seconds = 86400
    elif timestamp.endswith('W'):  # Week
        period_seconds = 604800
    elif timestamp.endswith('M'):  # Month
        period_seconds = 2628000  # Approximate
    elif timestamp.endswith('Q'):  # Quarter
        period_seconds = 7884000  # Approximate
    elif timestamp.endswith('Y'):  # Year
        period_seconds = 31536000
    else:
        raise ValueError("Unsupported timestamp format")

    # Function to convert time to radians based on the timestamp
    def time_to_radians(timestamp, dt, period_seconds):
        if timestamp.endswith('H'):
            # For hourly cycles, use the minutes and seconds
            total_seconds = (dt.minute * 60) + dt.second
        elif timestamp.endswith('D'):
            # For daily cycles, use the full time of day
            total_seconds = (dt.hour * 3600) + (dt.minute * 60) + dt.second
        elif timestamp.endswith('W'):
            # For weekly cycles, include the day of the week
            day_of_week = dt.weekday()  # Monday is 0, Sunday is 6
            total_seconds = (day_of_week * 86400) + (dt.hour * 3600) + (dt.minute * 60) + dt.second
        elif timestamp.endswith('M'):
            # For monthly cycles, include the day of the month
            day_of_month = dt.day - 1  # First of the month is 0
            total_seconds = (day_of_month * 86400) + (dt.hour * 3600) + (dt.minute * 60) + dt.second
        elif timestamp.endswith('Q'):
            # For quarterly cycles, include the day of the quarter
            # Assuming quarters start in January, April, July, October
            month_of_quarter = (dt.month - 1) % 3
            day_of_quarter = (month_of_quarter * 30) + (dt.day - 1)  # Approximation
            total_seconds = (day_of_quarter * 86400) + (dt.hour * 3600) + (dt.minute * 60) + dt.second
        elif timestamp.endswith('Y'):
            # For yearly cycles, include the day of the year
            day_of_year = dt.timetuple().tm_yday - 1  # January 1 is 0
            total_seconds = (day_of_year * 86400) + (dt.hour * 3600) + (dt.minute * 60) + dt.second
        else:
            raise ValueError("Unsupported timestamp format")

        return 2 * np.pi * (total_seconds % period_seconds) / period_seconds

    # Apply the function to each timestamp in the index
    radians = df.index.map(lambda dt: time_to_radians(timestamp, dt, period_seconds))

    # Calculate sine and cosine
    df[f'sin_time_{timestamp}'] = np.sin(radians)
    df[f'cos_time_{timestamp}'] = np.cos(radians)

    return df
