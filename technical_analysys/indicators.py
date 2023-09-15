import numpy as np
import random
import warnings
from tqdm import tqdm

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
