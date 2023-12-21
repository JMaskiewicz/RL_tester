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

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators
from data.edit import normalize_data, standardize_data

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

df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Define the time index for the dataset
df.reset_index(inplace=True)
df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days

target = 'Close_EURUSD'

static_categoricals = []
time_varying_known_reals = ['time_idx']
time_varying_unknown_reals = [['Close_EURUSD', 'Close_USDJPY', 'Close_EURJPY', 'ATR_24_EURUSD']]

df['Date'] = pd.to_datetime(df['Date'])
start_date = '2016-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[df['Date'] < validation_date]
df_validation = df[(df['Date'] >= validation_date) & (df['Date'] < test_date)]
df_test = df[df['Date'] >= test_date]

training = TimeSeriesDataSet(
    df_train,
    time_idx="time_idx",
    target=target)

validation = TimeSeriesDataSet(
    df_validation,
    time_idx="time_idx",
    target=target)

test = TimeSeriesDataSet(
    df_test,
    time_idx="time_idx",
    target=target)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        energy = self.projection(encoder_outputs)  # (batch_size, sequence_length, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch_size, sequence_length)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_size)
        return outputs, weights

class SimpleTemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, num_outputs, hidden_size, num_layers):
        super(SimpleTemporalFusionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        attention_out, _ = self.attention(lstm_out)
        output = self.fc(attention_out)
        return output


# Convert datasets to dataloaders
batch_size = 32
train_loader = DataLoader(training.to_dataloader(train=True, batch_size=batch_size, num_workers=0))
val_loader = DataLoader(validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0))
test_loader = DataLoader(test.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0))

num_epochs = 100

# Get parameters from the training dataset
dataset_parameters = training.get_parameters()
input_size = len(time_varying_known_reals) + len(time_varying_unknown_reals)

# Set other model parameters
num_outputs = 1  # You're predicting a single value ('Close_EURUSD')
hidden_size = 64  # Example value, you may tune this
num_layers = 2    # Example value, you may tune this

# Initialize the model with these parameters
model = SimpleTemporalFusionTransformer(input_size, num_outputs, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

first_batch = next(iter(train_loader))
print(first_batch)

# Training loop
for epoch in tqdm(range(num_epochs)):
    # Training phase
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        x, y = batch["x"], batch["y"]
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch["x"], batch["y"]
            outputs = model(x)
            val_loss = criterion(outputs, y)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

# Testing phase (after all epochs are done)
model.eval()
total_test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        x, y = batch["x"], batch["y"]
        outputs = model(x)
        test_loss = criterion(outputs, y)
        total_test_loss += test_loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")