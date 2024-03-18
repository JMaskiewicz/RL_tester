"""
Model description:
    - Input: 1D array of the last 20 closing prices
    - Output: 1D array of the last 20 closing prices
    - Action: 1, 0, -1


"""

# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from backtest.backtest_functions.other.backtest import Strategy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

from data.function.load_data import load_data

# set random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.epsilon_exponential_decay = 0.999
        self.variables = variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #  self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.action_space = action_space
        self.look_ahead = 1

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.float().to(self.device)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        return self.action_space[torch.argmax(act_values[0]).item()]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.from_numpy(next_state).float().to(self.device)
                target = reward + self.gamma * torch.max(self.model(next_state).detach()).item()

            state = torch.from_numpy(state).float().to(self.device)
            target_f = self.model(state)
            action = torch.tensor([action], device=self.device)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def calculate_discounted_reward(self, current_time, df_train):
        discounted_reward = 0.0
        for k in range(self.look_ahead):
            if current_time + k + 1 >= len(df_train):
                break
            reward = df_train.iloc[current_time + k + 1]['Close'] - df_train.iloc[current_time + k]['Close']
            discounted_reward += (self.gamma ** k) * reward
        return discounted_reward

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Example usage
currencies = ['EURUSD']
variables = ['Open', 'High', 'Low', 'Close']
df = load_data(currencies=currencies, timestamp_x='1D')
df = df.dropna()
start_date = '2010-01-01'
split_date = '2020-01-01'
df_train = df[start_date:split_date]

df_test = df[split_date:]

episodes = 20
action_space = np.arange(-1, 2, 1)
action_size = len(action_space)
batch_size = 4096
look_back = 20  # Number of previous observations to include
state_size = len(variables) * look_back
agent = DQNAgent(state_size, action_size)

all_total_rewards = []
episode_durations = []

for e in range(episodes):
    start_time = time.time()

    total_reward = 0
    for t in tqdm(range(look_back, len(df_train) - 1)):
        state = df_train.iloc[t - look_back:t][variables].values.flatten().reshape(1, -1)
        action = agent.act(state)

        next_state = df_train.iloc[t - look_back + 1:t + 1][variables].values.flatten().reshape(1, -1)
        reward = agent.calculate_discounted_reward(t, df_train) * action
        done = t == len(df_train) - 2

        agent.remember(state, action, reward, next_state, done)

        # Perform experience replay if enough memory is gathered
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

        if done & (e % 20 == 0):
            print(f"Episode: {e}/{episodes}")
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")

    episode_duration = time.time() - start_time  # Calculate duration
    episode_durations.append(episode_duration)
    all_total_rewards.append(total_reward)


import matplotlib.pyplot as plt

plt.plot(all_total_rewards, marker='o')  # Adds a circle marker at each data point
plt.grid(True)  # Adds a grid to the plot
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.show()






# TEST
# Initialize an empty list to store the predicted actions
predicted_actions = []

# Loop over the DataFrame to make predictions for each state
for i in range(look_back, len(df_test)):
    # Prepare the state from the current position minus look_back
    state = df_test.iloc[t - look_back:t][variables].values.flatten().reshape(1, -1)
    action = agent.act(state)

    # Append the predicted action to the list
    predicted_actions.append(action)

# Since we start predicting after 'look_back' rows, we need to prepend NaNs or a placeholder
# to align the length of the predictions with the DataFrame
padding = [None] * look_back
full_predictions = padding + predicted_actions

# Add the predicted actions to the DataFrame as a new column
df_test['Predicted_Action'] = full_predictions

# TRAIN
predicted_actions = []

for i in range(look_back, len(df_train)):
    state = df_train.iloc[t - look_back:t][variables].values.flatten().reshape(1, -1)
    action = agent.act(state)
    predicted_actions.append(action)

padding = [None] * look_back
full_predictions = padding + predicted_actions

df_train['Predicted_Action'] = full_predictions

class SimpleDQN(Strategy):
    def __init__(self, currencies, agent, look_back, leverage=1.0, provision=0.0001, starting_capital=10000):
        super().__init__(currencies, leverage, provision, starting_capital)
        self.agent = agent
        self.look_back = look_back

    def calculate_new_positions(self, df, current_index):
        if isinstance(current_index, pd.Timestamp):
            current_index = df.index.get_loc(current_index)
        new_positions = {}
        for currency in self.currencies:
            # Ensure the current_index is within the bounds of the dataframe
            if current_index >= self.look_back:
                state = df[variables].iloc[current_index - self.look_back:current_index].values.flatten().reshape(1, -1)
                action = self.agent.act(state)

                # Get the available capital at the current index
                available_capital = df.loc[df.index[current_index], 'Available_Capital'].iloc[0]

                # Using available capital for long or short, and none for holding
                if action == 1:
                    # Going long with available capital
                    new_positions[currency] = available_capital
                elif action == -1:
                    # Going short with available capital
                    new_positions[currency] = -available_capital
                else:
                    # Doing nothing, no position
                    new_positions[currency] = 0
            else:
                new_positions[currency] = 0

        return new_positions

# Usage Example
DQN_strategy = SimpleDQN(currencies, agent=agent, look_back=look_back, starting_capital=10000)
df_train = DQN_strategy.backtest(df_train)

report_df_positions = DQN_strategy.generate_report(df_train)
report = DQN_strategy.generate_extended_report()
DQN_strategy.display_summary_as_table(df_train, extended_report=True)

# Plotting
from backtest.plots.plot import plot_financial_data

plot_financial_data(df_train, DQN_strategy, currencies, volatility='garman_klass_volatility', n=50)