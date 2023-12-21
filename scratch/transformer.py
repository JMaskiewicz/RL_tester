import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_forecasting.models import TemporalFusionTransformer
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators
from data.edit import normalize_data, standardize_data

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output):
        energy = self.projection(lstm_output)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class SimpleTFT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(SimpleTFT, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        attention_out, _ = self.attention(lstm_out)
        output = self.fc(attention_out)
        return output


df = load_data(['EURUSD', 'USDJPY', 'EURJPY'], '1D')

indicators = [
    {"indicator": "ATR", "mkf": "EURUSD", "length": 24},
]
add_indicators(df, indicators)
df = df.dropna()

variables = [
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "EURJPY"), "edit": "normalize"},
    {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
]
tradable_markets = 'EURUSD'

look_back = 50
max_prediction_length = 10
df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Define the time index for the dataset
df.reset_index(inplace=True)
df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days

target = 'Close_EURUSD'

# Static and time-varying features
# You need to specify which features are known/unknown and static/dynamic.
# Adjust the following lists based on your dataset and prediction task.
static_categoricals = []
time_varying_known_reals = ['time_idx']
time_varying_unknown_reals = [col for col in df.columns if col not in ['Date', 'time_idx', target]]

# Define training cutoff to split the data (you already have this part)
training_cutoff = df['time_idx'].max() - max_prediction_length
df['Currency'] = df['Close_EURUSD'].apply(lambda x: 'EURUSD')


df['Date'] = pd.to_datetime(df['Date'])
start_date = '2016-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[df['Date'] < validation_date]
df_validation = df[(df['Date'] >= validation_date) & (df['Date'] < test_date)]
df_test = df[df['Date'] >= test_date]

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader
import torch

# Simplified TimeSeriesDataSet
training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target="Close_EURUSD",
    group_ids=["Currency"],
    max_encoder_length=30,
    max_prediction_length=7,
    static_categoricals=["Currency"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Close_EURUSD"],
    target_normalizer=GroupNormalizer(groups=["Currency"], transformation="softplus"),
    allow_missing_timesteps=True
)

# DataLoader
train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
# Simplified TimeSeriesDataSet
validation = TimeSeriesDataSet(
    df_validation,
    time_idx="time_idx",
    target="Close_EURUSD",
    group_ids=["Currency"],
    max_encoder_length=30,
    max_prediction_length=7,
    static_categoricals=["Currency"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Close_EURUSD"],
    target_normalizer=GroupNormalizer(groups=["Currency"], transformation="softplus"),
    allow_missing_timesteps=True
)

# DataLoader
validation_loader = validation.to_dataloader(train=True, batch_size=32, num_workers=0)

# Simplified TimeSeriesDataSet
test = TimeSeriesDataSet(
    df_test,
    time_idx="time_idx",
    target="Close_EURUSD",
    group_ids=["Currency"],
    max_encoder_length=30,
    max_prediction_length=7,
    static_categoricals=["Currency"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["Close_EURUSD"],
    target_normalizer=GroupNormalizer(groups=["Currency"], transformation="softplus"),
    allow_missing_timesteps=True
)

# DataLoader
test_loader = test.to_dataloader(train=True, batch_size=32, num_workers=0)

# Testing DataLoader
try:
    for i, batch in enumerate(train_loader):
        print(batch)
        print(f"Batch {i}: {batch['x']}, {batch['y']}")
        if i >= 2:  # Limit to first few batches
            break
except Exception as e:
    print(f"Error iterating over DataLoader: {e}")

input_size = 2  # time_idx and Close_EURUSD
num_outputs = 1  # Predicting Close_EURUSD
hidden_size = 64  # A starting point, adjust based on performance
num_layers = 2  # A starting point, can be adjusted

# Example training loop
num_epochs = 5
model = TemporalFusionTransformer(input_size, num_outputs, hidden_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in tqdm(range(num_epochs)):
    # Training Phase
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        x, y_batch = batch
        y = y_batch[0]  # Assuming the actual targets are the first element of the tuple
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            x, y_batch = batch
            y = y_batch[0]
            outputs = model(x)
            val_loss = criterion(outputs, y)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(validation_loader)

    # Print epoch results
    print(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

# Testing Phase (after all training epochs)
model.eval()
total_test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        x, y_batch = batch
        y = y_batch[0]
        outputs = model(x)
        test_loss = criterion(outputs, y)
        total_test_loss += test_loss.item()
avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")