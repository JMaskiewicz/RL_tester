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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from data.function.load_data import load_data, load_data_long_format
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns
from data.edit import normalize_data, standardize_data

if __name__ == '__main__':
    currencies = ['EURUSD', 'USDJPY', 'EURJPY']
    df = load_data(currencies, '1D')

    indicators = [
        {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
        {"indicator": "RSI", "mkf": 'USDJPY', "length": 14},
        {"indicator": "RSI", "mkf": 'EURJPY', "length": 14},
    ]

    add_indicators(df, indicators)

    for currency in currencies:
        if ("RSI_14", currency) in df.columns:
            df[("RSI_14", currency)] = df[("RSI_14", currency)] / 100

    df = df.dropna()
    start_date = '2013-01-01'
    validation_date = '2021-01-01'
    test_date = '2022-01-01'
    df_train = df[start_date:validation_date]
    df_validation = df[validation_date:test_date]
    df_test = df[test_date:]

    look_back = 5

    # Reset the index of df_train to make 'Date' a column and create a numeric 'time_idx'
    df_train = df_train.reset_index()
    df_train['time_idx'] = pd.factorize(df_train['Date'])[0]

    # Flatten multi-index columns if necessary
    df_train.columns = ['_'.join(col).strip() for col in df_train.columns.values]
    df_train = df_train.rename(columns={'time_idx_': 'time_idx', 'Date_': 'Date'})

    # Specify the target columns for each currency's 'Close' price
    target = 'Close_EURUSD'
    df_train['group_id'] = 'EURUSD'

    print(df_train.head())

    # Create the TimeSeriesDataSet
    df_training = TimeSeriesDataSet(
        df_train,
        time_idx='time_idx',
        target=target,
        group_ids=["group_id"],
        max_encoder_length=look_back,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        static_reals=["RSI_14_EURUSD"],
        time_varying_known_reals=[],
        time_varying_unknown_reals=[
            "Close_EURUSD",
            "Close_USDJPY",
            "Close_EURJPY",
        ],
    )

    batch_size = 32

    # Print to verify setup
    print(df_training)

    # pass the dataset to a dataloader
    train_dataloader = df_training.to_dataloader(train=True, batch_size=batch_size, num_workers=27, persistent_workers=True)

    #load the first batch
    x, y = next(iter(train_dataloader))
    print(x['encoder_target'])
    print(x['groups'])
    print('\n')
    print(x['decoder_target'])

    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")


    class TFTLightningModule(pl.LightningModule):
        def __init__(self, tft_model):
            super().__init__()
            self.model = tft_model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = self.model.loss(y, y_hat)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu',
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger],
        logger=logger)

    tft_model = TemporalFusionTransformer.from_dataset(
        df_training,
        learning_rate=0.001,
        hidden_size=160,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=160,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4)

    # Wrap the TFT model with the LightningModule wrapper
    tft_lightning = TFTLightningModule(tft_model)

    # Train using PyTorch Lightning's Trainer
    trainer.fit(
        tft_lightning,
        train_dataloaders=train_dataloader
    )

    '''class TFTModel(LightningModule):
        def __init__(self, tft_model):
            super().__init__()
            self.model = tft_model
    
        def forward(self, x):
            return self.model(x)
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = self.model.loss(y, y_hat)
            self.log('train_loss', loss)
            return loss
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.03)
            return optimizer
    
    # Initialize your TemporalFusionTransformer as before
    tft_model = TemporalFusionTransformer.from_dataset(
        training_set,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=3,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    batch_size = 128'''

    '''train_dataloader = training_set.to_dataloader(train=True, batch_size=batch_size, num_workers=27, persistent_workers=True)
    
    # Wrap the TFT model into the LightningModule
    tft_lightning_model = TFTModel(tft_model)
    
    # Now use the wrapped model with the Trainer
    trainer = Trainer(max_epochs=10)'''

    '''    
    for idx in range(len(training_set)):
        sample = training_set[idx]
        print(sample)
        if idx == 0:
            break
    
    for batch in train_dataloader:
        x, y = batch
        if x is None or y is None:
            print("Found a None value in the batch")
            break
    '''

    '''trainer.fit(tft_lightning_model, train_dataloader)
    
    df_test = df_test.reset_index()
    df_test['time_idx'] = pd.factorize(df_test['Date'])[0]
    df_test.columns = ['_'.join(col).strip() for col in df_test.columns.values]
    df_test = df_test.rename(columns={'time_idx_': 'time_idx', 'Date_': 'Date'})
    df_test['group_id'] = 'all'
    
    # Create TimeSeriesDataSet for test data
    test_set = TimeSeriesDataSet(
        df_test,
        time_idx='time_idx',
        target=target,
        group_ids=["group_id"],
        max_encoder_length=look_back,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_unknown_reals=[
            "Close_EURUSD",
            "Close_USDJPY",
            "Close_EURJPY",
            "RSI_14_EURUSD",
            "RSI_14_USDJPY",
            "RSI_14_EURJPY",
        ],
    )
    
    # Create DataLoader from test_set
    test_dataloader = test_set.to_dataloader(train=False, batch_size=batch_size, num_workers=27)
    
    # Make predictions
    predictions = []
    
    for batch in test_dataloader:
        x, y = batch
        predictions.append(tft_lightning_model(x))'''

    print('end')