"""
PPO 2.1

# TODO LIST
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- try TFT transformer (Temporal Fusion Transformers transformer time series)

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
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns
from data.edit import normalize_data, standardize_data
import backtest.backtest_functions.functions as BF

# TODO add proper backtest function
def generate_predictions_and_backtest(df, agent, tradable_market, look_back, variables, provision=0.0001, initial_balance=10000, leverage=1):
    # Switch to evaluation mode
    agent.q_policy.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        # Create a validation environment
        validation_env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                                   tradable_markets=tradable_market, provision=provision,
                                                   initial_balance=initial_balance, leverage=leverage)

        # Generate Predictions
        predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
        for validation_observation in range(len(df) - validation_env.look_back):
            observation = validation_env.reset()
            action = agent.choose_best_action(observation)  # choose_best_action should be used for inference
            predictions_df.iloc[validation_observation + validation_env.look_back] = action

        # Merge with original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action']

        # Backtesting
        balance = initial_balance
        current_position = 0  # Neutral position
        total_reward = 0  # Initialize total reward
        number_of_trades = 0

        for i in range(look_back, len(df_with_predictions)):
            action = df_with_predictions['Predicted_Action'].iloc[i]
            current_price = df_with_predictions[('Close', tradable_market)].iloc[i - 1]
            next_price = df_with_predictions[('Close', tradable_market)].iloc[i]

            # Calculate log return
            log_return = np.log(next_price / current_price) if current_price != 0 else 0
            reward = log_return * (action - 1) * leverage  # Adjust action for {0, 1, 2} -> {-1, 0, 1}

            # Calculate provision cost if the action changes the position
            if action != current_position:
                reward -= provision
                number_of_trades += 1

            # Update the balance
            balance *= (1 + reward)

            # Update the current position
            current_position = action

            # Accumulate the reward
            total_reward += reward

    # Switch back to training mode
    agent.q_policy.train()

    return balance, total_reward, number_of_trades

def backtest_wrapper(df, agent, mkf, look_back, variables, provision, initial_balance, leverage):
    return generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision, initial_balance, leverage)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class QNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def clear_memory(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros_like(self.state_memory)
        self.new_state_memory = np.zeros_like(self.new_state_memory)
        self.action_memory = np.zeros_like(self.action_memory)
        self.reward_memory = np.zeros_like(self.reward_memory)
        self.terminal_memory = np.zeros_like(self.terminal_memory)


class DDQN_Agent:
    def __init__(self, input_dims, n_actions, epochs=1, mini_batch_size=256, gamma=0.99, alpha=0.001, epsilon=1.0, epsilon_dec=1e-5, epsilon_end=0.01, mem_size=100000, batch_size=64, replace=1000):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO repair cuda
        self.device = torch.device("cpu")
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.alpha = alpha
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, (input_dims,), n_actions)
        self.q_policy = QNetwork(input_dims, n_actions).to(self.device)
        self.q_target = QNetwork(input_dims, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_policy.parameters(), lr=self.alpha)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)


    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        for epoch in range(self.epochs):
            states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

            num_mini_batches = max(self.batch_size // self.mini_batch_size, 1)

            for mini_batch in range(num_mini_batches):
                start = mini_batch * self.mini_batch_size
                end = min((mini_batch + 1) * self.mini_batch_size, self.batch_size)

                mini_states = torch.tensor(states[start:end], dtype=torch.float).to(self.device)
                mini_actions = actions[start:end]
                mini_rewards = torch.tensor(rewards[start:end], dtype=torch.float).to(self.device)
                mini_states_ = torch.tensor(states_[start:end], dtype=torch.float).to(self.device)
                mini_dones = dones[start:end]

                self.optimizer.zero_grad()

                q_pred = self.q_policy(mini_states).gather(1, torch.tensor(mini_actions).unsqueeze(-1).to(
                    self.device)).squeeze(-1)
                q_next = self.q_target(mini_states_).detach()
                q_eval = self.q_policy(mini_states_).detach()

                max_actions = torch.argmax(q_eval, dim=1)
                q_next[mini_dones] = 0.0
                q_target = mini_rewards + self.gamma * q_next.gather(1, max_actions.unsqueeze(-1)).squeeze(-1)

                loss = F.mse_loss(q_pred, q_target)
                loss.backward()
                self.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            actions = self.q_policy(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def get_action_q_values(self, observation):
        """
        Returns the Q-values of each action for a given observation.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        with torch.no_grad():
            q_values = self.q_policy(state)

        return q_values.cpu().numpy()

    def choose_best_action(self, observation):
        """
        Selects the best action based on the highest Q-value without exploration.
        """
        q_values = self.get_action_q_values(observation)
        best_action = np.argmax(q_values)
        return best_action

    def get_epsilon(self):
        return self.epsilon

    # TODO
    def get_action_probabilities(self, observation):
        """
        Returns the probabilities of each action for a given observation.
        This is not quite forward as DDQN uses Q values to choose the best action, not probabilities.
        There is need to somehow convert Q values to probabilities.
        """
        pass


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000, leverage=1):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0  # Initial position: 0 (neutral)
        self.variables = variables if variables is not None else []
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage

        # Define action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.Discrete(3)

        # Define observation space based on the look_back period and the number of variables
        input_dims = self.calculate_input_dims()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(input_dims,), dtype=np.float32)

        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        return num_variables * self.look_back  # Total dimensions based on variables and look_back period

    def reset(self):
        self.current_step = self.look_back  # Start from the look_back position
        self.balance = self.initial_balance
        self.current_position = 0  # Reset to neutral position
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        start = self.current_step - self.look_back
        end = self.current_step

        # Generate observation based on the selected variables and look_back period
        scaled_observation = []
        for variable in self.variables:
            data = self.df[variable['variable']].iloc[start:end].values
            if variable['edit'] == 'standardize':
                scaled_data = (data - np.mean(data)) / np.std(data)
            elif variable['edit'] == 'normalize':
                scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            else:
                scaled_data = data  # No scaling applied

            scaled_observation.extend(scaled_data)

        return np.array(scaled_observation)

    def step(self, action):
        # Map action to position change (-1: sell, 0: hold, +1: buy)
        action_mapping = {0: -1, 1: 0, 2: 1}
        mapped_action = action_mapping[action]

        # Get current and next price from the tradable market
        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step - 1]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]

        # Calculate log return and reward based on the action taken
        log_return = np.log(next_price / current_price) if current_price > 0 else 0
        reward = log_return * mapped_action * self.leverage  # Apply leverage

        # Apply trading provision cost if action changes the position
        if mapped_action != self.current_position:
            reward -= self.provision

        # Update balance, position, and step
        self.balance *= (1 + reward)
        self.current_position = mapped_action
        self.current_step += 1

        # Check if the episode is done
        self.done = self.current_step >= len(self.df)

        return self._next_observation(), reward, self.done, {}

# Example usage
# Stock market variables
df = load_data(['EURUSD', 'USDJPY', 'EURJPY'], '1D')

indicators = [
    {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
    {"indicator": "ATR", "mkf": "EURUSD", "length": 24},
    {"indicator": "MACD", "mkf": "EURUSD"},
    {"indicator": "Stochastic", "mkf": "EURUSD"},]

add_indicators(df, indicators)

df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")]/100
df = df.dropna()
start_date = '2014-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[start_date:validation_date]
df_validation = df[validation_date:test_date]
df_test = df[test_date:]
variables = [
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "EURJPY"), "edit": "normalize"},
    {"variable": ("RSI_14", "EURUSD"), "edit": "None"},
    {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
]
tradable_markets = 'EURUSD'
window_size = '10Y'
starting_balance = 10000
look_back = 10
provision = 0.001  # 0.001, cant be too high as it would not learn to trade

# Training parameters
batch_size = 2048
epochs = 50  # 40
mini_batch_size = 128
leverage = 1
weight_decay = 0.0005 # TODO add
l1_lambda = 1e-5 # TODO add
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
agent = DDQN_Agent(input_dims=env.observation_space.shape[0],
                   n_actions=env.action_space.n,
                   epochs=epochs,
                   mini_batch_size=mini_batch_size,
                   alpha=0.001,
                   gamma=0.95,
                   epsilon=1.0,
                   epsilon_dec=1e-5,
                   epsilon_end=0.01,
                   mem_size=100000,
                   batch_size=64,
                   replace=1000)

num_episodes = 1000  # 100

total_rewards = []
episode_durations = []
total_balances = []
episode_probabilities = {'train': [], 'validation': [], 'test': []}

index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
columns = ['Final Balance', 'Dataset Index']
backtest_results = pd.DataFrame(index=index, columns=columns)

# Rolling DF
rolling_datasets = rolling_window_datasets(df_train, window_size=window_size,  look_back=look_back)
dataset_iterator = cycle(rolling_datasets)

for episode in tqdm(range(num_episodes)):
    window_df = next(dataset_iterator)
    dataset_index = episode % len(rolling_datasets)

    print(f"\nEpisode {episode + 1}: Learning from dataset with Start Date = {window_df.index.min()}, End Date = {window_df.index.max()}, len = {len(window_df)}")
    # Create a new environment with the randomly selected window's data
    env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    observation = env.reset()
    done = False
    cumulative_reward = 0
    start_time = time.time()
    initial_balance = env.balance

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        cumulative_reward += reward

        # Check if enough data is collected or if the dataset ends
        if agent.memory.mem_cntr >= agent.batch_size or done:
            agent.learn()
            agent.memory.clear_memory()

    # Backtesting in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        validation_future = executor.submit(backtest_wrapper, df_validation, agent, 'EURUSD', look_back, variables,
                                            provision, starting_balance, leverage)
        test_future = executor.submit(backtest_wrapper, df_test, agent, 'EURUSD', look_back, variables, provision,
                                      starting_balance, leverage)

        # Retrieve results
        validation_balance, validation_total_rewards, validation_number_of_trades = validation_future.result()
        test_balance, test_total_rewards, test_number_of_trades = test_future.result()

    backtest_results.loc[(episode, 'validation'), 'Final Balance'] = validation_balance
    backtest_results.loc[(episode, 'test'), 'Final Balance'] = test_balance
    backtest_results.loc[(episode, 'validation'), 'Final Reward'] = validation_total_rewards
    backtest_results.loc[(episode, 'test'), 'Final Reward'] = test_total_rewards
    backtest_results.loc[(episode, 'validation'), 'Number of Trades'] = validation_number_of_trades
    backtest_results.loc[(episode, 'test'), 'Number of Trades'] = test_number_of_trades
    backtest_results.loc[(episode, 'validation'), 'Dataset Index'] = dataset_index
    backtest_results.loc[(episode, 'test'), 'Dataset Index'] = dataset_index

    # results
    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(cumulative_reward)
    episode_durations.append(episode_time)
    total_balances.append(env.balance)

    print(f"\nCompleted learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
    print("-----------------------------------")


print(backtest_results)
# Plotting the results after all episodes
import matplotlib.pyplot as plt

# Plotting the results after all episodes
plt.plot(total_rewards)
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

plt.plot(episode_durations, color='red')
plt.title('Episode Duration Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Episode Duration')
plt.show()

plt.plot(total_balances, color='green')
plt.title('Total Balance Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Balance')
plt.show()

print('end')
