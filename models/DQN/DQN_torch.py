"""
Model description:
    - Input: array of the last n closing prices
    - Output: 1D array of the last n closing prices
    - Action: 1, 0, -1

DQN agent:
    - looking only on next reward (gamma = 0)

Backtesting
    - playing with full capital
    - no leverage
"""

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data.function.load_data import load_data


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class DQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, architecture):
        super(DQNetwork, self).__init__()
        layers = []
        for units in architecture:
            layers.append(nn.Linear(state_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            state_dim = units

        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(architecture[-1], num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay_steps, epsilon_exponential_decay, replay_capacity, architecture, batch_size, variables):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.batch_size = batch_size

        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = DQNetwork(state_dim, num_actions, architecture).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.experience = deque(maxlen=replay_capacity)

        self.variables = variables

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(-1, 2)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.network(state)
        action = np.argmax(q_values.cpu().detach().numpy()) - 1
        return action

    def memorize_transition(self, s, a, r):
        self.experience.append((s, a, r))

    def replay(self):
        if len(self.experience) < self.batch_size:
            return

        # Sample a minibatch from the memory
        minibatch = random.sample(self.experience, self.batch_size)
        states, actions, rewards = map(np.array, zip(*minibatch))

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        # Compute Q values for current states
        q_values = self.network(states)
        q_values = q_values.squeeze(1)  # Remove the extra dimension

        # Targets = r (gamma=0)
        targets = rewards

        q_values = q_values.gather(1, (actions + 1).unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, targets).to(self.device)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Epsilon update for exploration-exploitation tradeoff
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_exponential_decay


# Example usage
currencies = ['EURUSD']
df = load_data(currencies=currencies, timestamp_x='1D')
df = df.dropna()
start_date = '2005-01-01'
split_date = '2020-01-01'
df_train = df[start_date:split_date]
df_test = df[split_date:]


variables = ['Close']
episodes = 1100
action_size = len(np.arange(-1, 2, 1))
look_back = 10  # Number of previous observations to include
state_size = len(variables) * look_back

agent = DQNAgent(
    state_dim=state_size,
    num_actions=action_size,
    learning_rate=0.01,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=1000000,
    epsilon_exponential_decay=0.9999,
    replay_capacity=int(1e6),
    architecture=(512, 512, 512, 512),
    batch_size=2048,
    variables=variables
)


def train_agent(agent, df_train, episodes, look_back):
    total_rewards = []
    episode_durations = []

    for episode in range(episodes):
        start_time = time.time()  # Start time of the episode

        state = get_state(df_train, look_back)
        total_reward = 0
        step = 0
        while True:
            if step >= look_back - 1:  # Ensure complete look-back window
                action = agent.epsilon_greedy_policy(state)
                reward, done = get_reward(df_train, state, action, step)
                agent.memorize_transition(state, action, reward)
                total_reward += reward
                agent.replay()
                if done:
                    break

            step += 1

        end_time = time.time()  # End time of the episode
        episode_time = end_time - start_time  # Calculate the duration

        total_rewards.append(total_reward)
        episode_durations.append(episode_time)

        if episode % 10 == 0:  # Adjust the frequency of printing if needed
            print('Episode: ', episode + 1)
            print("Current Epsilon: ", agent.epsilon)
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Duration: {step}, Time: {episode_time:.2f} seconds")
            print('----')

    return total_rewards, episode_durations


def get_state(df, look_back):
    data = df.iloc[:look_back][variables].values.flatten().reshape(1, -1)
    return data

def get_reward(df, state, action, t):
    if t < len(df) - 1:
        price_diff = (df.iloc[t + 1]['Close'] - df.iloc[t]['Close']) / df.iloc[t]['Close'] * 100
        reward = action * price_diff.iloc[0]
    else:
        reward = 0
    done = t == len(df) - 1
    return reward, done

all_total_rewards, episode_durations = train_agent(agent, df_train, episodes, look_back)

# Plotting the results after all episodes
plt.plot(all_total_rewards)
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


def calculate_win_rate(df, currency):
    # Calculate price change
    df[('Price_Change', currency)] = df[('Close', currency)].diff()

    # Define win and loss conditions
    win_condition = ((df[('Price_Change', currency)] > 0) & (df[('Action', currency)] == 1)) | \
                    ((df[('Price_Change', currency)] < 0) & (df[('Action', currency)] == -1))

    loss_condition = ((df[('Price_Change', currency)] > 0) & (df[('Action', currency)] == -1)) | \
                     ((df[('Price_Change', currency)] < 0) & (df[('Action', currency)] == 1))

    # Count wins and losses
    wins = df[win_condition].shape[0]
    losses = df[loss_condition].shape[0]
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    number_of_trades = wins + losses

    return win_rate, number_of_trades

def make_predictions_with_agent(agent, df, look_back, currency):
    predicted_actions = [None] * look_back  # Initialize with None for initial timestamps

    for t in tqdm(range(len(df) - look_back)):
        state = get_state(df[variables].iloc[t:t + look_back], look_back)
        action = agent.epsilon_greedy_policy(state)
        predicted_actions.append(action)

    # Ensure the predicted actions list matches the DataFrame length
    predicted_actions.extend([None] * (len(df) - len(predicted_actions)))

    # Add the predicted actions as a new column to the DataFrame
    # Adjust this part based on your DataFrame's structure
    df[('Action', currency)] = predicted_actions

    return df


predicted_df_train = make_predictions_with_agent(agent, df_train, look_back, currencies[0])
predicted_df_test = make_predictions_with_agent(agent, df_test, look_back, currencies[0])

win_rate_train,  number_of_trades_train = calculate_win_rate(predicted_df_train, currencies[0])
print(f"Win Rate train: {win_rate_train * 100:.2f}%")
print(f"Number of trades train: {number_of_trades_train}")
win_rate_test,  number_of_trades_test = calculate_win_rate(predicted_df_test, currencies[0])
print(f"Win Rate test: {win_rate_test * 100:.2f}%")
print(f"Number of trades test: {number_of_trades_test}")

from backtest.backtest_functions.backtest import Strategy

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
                action = agent.epsilon_greedy_policy(state)
                # Get the available capital at the current index
                available_capital = df.loc[df.index[current_index], 'Available_Capital'].iloc[0]

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
predicted_df_test = DQN_strategy.backtest(predicted_df_test)

report_df_positions = DQN_strategy.generate_report(predicted_df_test)
report = DQN_strategy.generate_extended_report()
DQN_strategy.display_summary_as_table(predicted_df_test, extended_report=True)

# Plotting
from backtest.plots.plot import plot_financial_data

plot_financial_data(predicted_df_test, DQN_strategy, currencies, volatility='garman_klass_volatility', n=100)

