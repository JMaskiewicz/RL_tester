import numpy as np
import random
import warnings
from tqdm import tqdm

from technical_analysys.indicators import RSI, EMA
from technical_analysys.volatility_functions import close_to_close_volatility, parkinson_volatility, garman_klass_volatility, rogers_satchell_volatility

def add_indicators(df, currency, **kwargs):
    """Add specified indicators to the DataFrame for the provided currency."""
    for indicator, length in kwargs.items():
        if indicator == 'RSI':
            df[f'{currency}_{indicator}_{length}', currency] = RSI(df['Close'][currency], length)
        elif indicator == 'EMA':
            df[f'{currency}_{indicator}_{length}', currency] = EMA(df['Close'][currency], length)
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