# library
import numpy as np

def close_to_close_volatility(df, currency, n):
    """
    Compute Close-to-Close Historical Volatility.
    """
    returns = np.log(df['Close', currency] / df['Close', currency].shift(1))
    volatility = returns.rolling(window=n).std()
    return volatility

def parkinson_volatility(df, currency, n):
    """
    Compute Parkinson's Volatility.
    """
    high_low_ratio = df['High', currency] / df['Low', currency]
    log_ratio = np.log(high_low_ratio) ** 2
    volatility = np.sqrt((1 / (4 * np.log(2))) * log_ratio).rolling(window=n).mean()  # Taking average after computing the daily values
    return volatility

def garman_klass_volatility(df, currency, n):
    """
    Compute Garman-Klass Volatility.
    """
    term1 = 0.5 * np.log(df['High', currency] / df['Low', currency]) ** 2
    term2 = 0.386 * np.log(df['Close', currency] / df['Open', currency]) ** 2
    volatility = np.sqrt(term1 - term2).rolling(window=n).mean()  # Taking average after computing the daily values
    return volatility

def rogers_satchell_volatility(df, currency, n):
    """
    Compute Rogers-Satchell Volatility.
    """
    term1 = np.log(df['High', currency] / df['Close', currency])
    term2 = np.log(df['Low', currency] / df['Close', currency])
    daily_volatility = np.sqrt(term1 * term2)
    volatility = daily_volatility.rolling(window=n).mean()
    return volatility