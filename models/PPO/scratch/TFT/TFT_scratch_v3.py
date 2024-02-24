"""
TFT network
"""
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import math
import gym
from gym import spaces
from itertools import cycle
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet
import torch
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting import QuantileLoss
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from data.function.load_data import load_data, load_data_long_format
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns
from data.edit import normalize_data, standardize_data

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def calculate_RSI(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI


# Load data
df = load_data_long_format(['EURUSD', 'USDJPY', 'EURJPY'], '1D')

# Calculate RSI and normalize
df['RSI_14'] = df.groupby('Currency')['Close'].transform(lambda x: calculate_RSI(x))
df["RSI_14"] = df["RSI_14"] / 100

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter data based on date ranges
start_date = '2017-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'

# Create a copy to avoid SettingWithCopyWarning
df_train = df[(df['Date'] >= start_date) & (df['Date'] < validation_date)].copy()
df_validation = df[(df['Date'] >= validation_date) & (df['Date'] < test_date)].copy()
df_test = df[df['Date'] >= test_date].copy()

# Process each DataFrame to avoid SettingWithCopyWarning
for df_subset in [df_train, df_validation, df_test]:
    df_subset['time_idx'] = df_subset.groupby('Currency').cumcount()
    df_subset.set_index(['Date', 'Currency'], inplace=True)

# Define parameters for TimeSeriesDataSet
look_back = 5

# Since group_ids cannot be empty, we use 'Currency' as the group ID
df_train = df_train.reset_index()
df_train = df_train.sort_values(['time_idx', 'Currency'])

print(df_train.head())

# Create the TimeSeriesDataSet for training
df_training = TimeSeriesDataSet(
    df_train,
    time_idx='time_idx',
    target="Close",
    group_ids=["Currency"],
    max_encoder_length=look_back,
    max_prediction_length=1,
    static_categoricals=["Currency"],
    static_reals=["RSI_14"],
    time_varying_known_reals=[],
    time_varying_unknown_reals=["Close"],
)

# Print to verify setup
print(df_training)

# pass the dataset to a dataloader
dataloader = df_training.to_dataloader(batch_size=1)

#load the first batch
x, y = next(iter(dataloader))
print(x['encoder_target'])
print(x['groups'])
print('\n')
print(x['decoder_target'])



