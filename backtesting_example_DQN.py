# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data.function.load_data import load_data
from technical_analysys.add_indicators import add_indicators, compute_volatility
from backtest.backtest_functions.backtest import Strategy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import time

from data.function.load_data import load_data

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.action_space = np.arange(-100, 101, 25)
        self.look_ahead = 10

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
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
            state = torch.from_numpy(state).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.tensor(reward).float().to(self.device)
            action_index = np.where(self.action_space == action)[0][0]

            self.optimizer.zero_grad()
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state)
            target_f[0][action_index] = target
            loss = nn.MSELoss()(target_f, target_f.clone().detach())
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_discounted_reward(self, current_time, df_train):
        discounted_reward = 0.0
        for k in range(self.look_ahead):
            if current_time + k + 1 >= len(df_train):
                break
            reward = np.log(df_train.iloc[current_time + k + 1]['Close']) - np.log(df_train.iloc[current_time + k]['Close'])
            discounted_reward += (self.gamma ** k) * reward
        return discounted_reward

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

# Example usage
currencies = ['EURUSD']
#df = load_data_rar(currencies=currencies, timestamp_x='1D')
df = load_data(currencies=currencies, timestamp_x='1D')
df = df.dropna()
start_date = '2020-06-01'
split_date = '2021-01-01'
df_train = df[start_date:split_date]
#df_train = df_train[['Close']]

df_test = df[split_date:]
#df_test = df_test[['Close']]

episodes = 2000
action_size = len(np.arange(-100, 101, 25))
batch_size = 64  # 32
look_back = 20  # Number of previous observations to include
state_size = 1 * look_back
agent = DQNAgent(state_size, action_size)

all_total_rewards = []
episode_durations = []

for e in range(episodes):
    start_time = time.time()
    print(f"Episode: {e}/{episodes}")
    total_reward = 0
    for t in tqdm(range(look_back, len(df_train) - 1)):
        state = df_train.iloc[t - look_back:t]['Close'].values.flatten().reshape(1, -1)
        action = agent.act(state)

        next_state = df_train.iloc[t - look_back + 1:t + 1]['Close'].values.flatten().reshape(1, -1)
        reward = agent.calculate_discounted_reward(t, df_train) * action
        done = t == len(df_train) - 2

        agent.remember(state, action, reward, next_state, done)

        # Perform experience replay if enough memory is gathered
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")

    episode_duration = time.time() - start_time  # Calculate duration
    episode_durations.append(episode_duration)
    all_total_rewards.append(total_reward)

# Assuming all_total_rewards is a list containing the total rewards of each episode
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(all_total_rewards, marker='o', linestyle='-')  # Plot the rewards
plt.title("Total Rewards per Episode")  # Title of the plot
plt.xlabel("Episode")  # X-axis label
plt.ylabel("Total Reward")  # Y-axis label
plt.grid(True)  # Show grid
plt.show()  # Display the plot

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
                state = df['Close'].iloc[current_index - self.look_back:current_index].values.flatten().reshape(1, -1)
                action = self.agent.act(state)
                new_positions[currency] = action
            else:
                new_positions[currency] = 0

        return new_positions


# Usage Example
DQN_strategy = SimpleDQN(currencies, agent=agent, look_back=look_back, starting_capital=10000)
df_test = DQN_strategy.backtest(df_test)

report_df_positions = DQN_strategy.generate_report(df_test)
report = DQN_strategy.generate_extended_report()
DQN_strategy.display_summary_as_table(df_test, extended_report=True)

# Plotting
from backtest.plots.plot import plot_financial_data

plot_financial_data(df_test, DQN_strategy, currencies, volatility='garman_klass_volatility', n=20)

# Usage Example
DQN_strategy = SimpleDQN(currencies, agent=agent, look_back=look_back, starting_capital=10000)
df_train = DQN_strategy.backtest(df_train)

report_df_positions = DQN_strategy.generate_report(df_train)
report = DQN_strategy.generate_extended_report()
DQN_strategy.display_summary_as_table(df_train, extended_report=True)

# Plotting
plot_financial_data(df_train, DQN_strategy, currencies, volatility='garman_klass_volatility', n=20)