import numpy as np
from tqdm import tqdm

from technical_analysys.indicators import rsi, simple_moving_average, average_true_range, macd, stochastic_oscillator, parabolic_sar
from technical_analysys.volatility_functions import close_to_close_volatility, parkinson_volatility, garman_klass_volatility, rogers_satchell_volatility

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