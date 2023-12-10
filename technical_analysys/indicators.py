import pandas as pd

def rsi(df, mkf, length=14):
    close_prices = df.xs(mkf, level='Currency', axis=1)['Close']
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def simple_moving_average(df, mkf, length=100):
    return df.xs(mkf, level='Currency', axis=1)['Close'].rolling(window=length, min_periods=1).mean()

def exponential_moving_average(df, mkf, length=14):
    close_prices = df.xs(mkf, level='Currency', axis=1)['Close']
    return close_prices.ewm(span=length, adjust=False, min_periods=1).mean()

def average_true_range(df, mkf, length=14):
    market_data = df.xs(mkf, level='Currency', axis=1)
    high = market_data['High']
    low = market_data['Low']
    close = market_data['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr

def macd(df, mkf, short_window=12, long_window=26, signal_window=9):
    market_data = df.xs(mkf, level='Currency', axis=1)
    short_ema = market_data['Close'].ewm(span=short_window, adjust=False, min_periods=1).mean()
    long_ema = market_data['Close'].ewm(span=long_window, adjust=False, min_periods=1).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False, min_periods=1).mean()
    return macd_line, signal_line

def stochastic_oscillator(df, mkf, k_window=14, d_window=3):
    market_data = df.xs(mkf, level='Currency', axis=1)
    low_min = market_data['Low'].rolling(window=k_window, min_periods=1).min()
    high_max = market_data['High'].rolling(window=k_window, min_periods=1).max()
    k_percent = 100 * ((market_data['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()
    return k_percent, d_percent

def parabolic_sar(df, mkf, af=0.02, af_max=0.2):
    market_data = df.xs(mkf, level='Currency', axis=1)
    high = market_data['High']
    low = market_data['Low']
    sar = low.rolling(2).min().copy()
    ep = high.rolling(2).max().copy()
    trend = 1
    af_value = af

    for i in range(2, len(market_data)):
        if trend == 1:
            sar.iloc[i] = max(sar.iloc[i - 1], high.iloc[i - 1], high.iloc[i - 2])
            if low.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep.iloc[i - 1]
                af_value = af
                continue
        else:
            sar.iloc[i] = min(sar.iloc[i - 1], low.iloc[i - 1], low.iloc[i - 2])
            if high.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep.iloc[i - 1]
                af_value = af
                continue
        if (trend == 1 and high.iloc[i] > ep.iloc[i - 1]) or (trend == -1 and low.iloc[i] < ep.iloc[i - 1]):
            ep.iloc[i] = high.iloc[i] if trend == 1 else low.iloc[i]
            af_value = min(af_value + af, af_max)
        sar.iloc[i] = sar.iloc[i - 1] + af_value * (ep.iloc[i - 1] - sar.iloc[i - 1])
    return sar
