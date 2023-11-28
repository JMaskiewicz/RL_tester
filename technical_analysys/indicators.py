import numpy as np
import random
import warnings
from tqdm import tqdm
import pandas as pd

def RSI(series, length=14):
    """Compute the RSI (Relative Strength Index) of the provided series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def EMA(series, length=14):
    """Compute the EMA (Exponential Moving Average) of the provided series."""
    ema = series.ewm(span=length, adjust=False).mean()
    return ema


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def calculate_stochastic_oscillator(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d = k.rolling(window=d_window).mean()
    return d - k

def calculate_ultosc(high, low, close):
    # This is a simplified version. The actual calculation may be more complex.
    bp = close - low
    tr = high - low
    avg_bp = bp.rolling(window=7).mean()
    avg_tr = tr.rolling(window=7).mean()
    ultosc = 100 * (avg_bp / avg_tr)
    return ultosc