"""
the newest version of Proximal Policy Optimization agent with Temporal Fusion Transformer

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

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns
from data.edit import normalize_data, standardize_data

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# TODO add backtest function
def generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision=0.0001, initial_balance=10000, leverage=1):
    # Create a validation environment
    validation_env = Trading_Environment_Basic(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=initial_balance, leverage=leverage)

    # Generate Predictions
    predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
    for validation_observation in range(len(df) - validation_env.look_back):
        observation = validation_env.reset(validation_observation)
        action = agent.choose_best_action(observation)
        predictions_df.iloc[validation_observation + validation_env.look_back] = action

    # Merge with original DataFrame
    df_with_predictions = df.copy()
    df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1

    # Backtesting
    balance = initial_balance
    current_position = 0  # -1 (sell), 0 (hold), 1 (buy)

    # keep track of number of trades
    number_of_trades = 0

    for i in range(1, len(df_with_predictions)):
        action = df_with_predictions['Predicted_Action'].iloc[i]
        current_price = df_with_predictions[('Close', mkf)].iloc[i - 1]
        next_price = df_with_predictions[('Close', mkf)].iloc[i]

        # Calculate log return
        log_return = math.log(next_price / current_price) if current_price != 0 else 0
        reward = 0

        if action == 1:  # Buying
            reward = log_return
        elif action == -1:  # Selling
            reward = -log_return

        # Calculate cost based on action and current position
        if action != current_position:
            if action == 0 or math.isnan(action):
                provision_cost = 0
            else:
                provision_cost = math.log(1 - 2 * provision)
                number_of_trades += 1
        else:
            provision_cost = 0
        reward += provision_cost

        # Update the balance
        balance *= math.exp(reward)

        # Update current position
        current_position = action

    return balance, number_of_trades


# TODO use deque instead of list
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.static_inputs = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        states = np.array(self.states)[indices]
        actions = np.array(self.actions)[indices]
        probs = np.array(self.probs)[indices]
        vals = np.array(self.vals)[indices]
        rewards = np.array(self.rewards)[indices]
        dones = np.array(self.dones)[indices]
        static_inputs = np.array(self.static_inputs)[indices]

        return states, actions, probs, vals, rewards, dones, static_inputs, batches

    def store_memory(self, state, action, probs, vals, reward, done, static_input):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(float(reward))
        self.dones.append(done)
        self.static_inputs.append(static_input)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.static_inputs = []

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(True),
            nn.Dropout(1/16),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(1/16),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(1/16),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_output):
        energy = self.projection(lstm_output)
        weights = F.softmax(energy, dim=1)
        outputs = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class LSTM_NetworkBase(nn.Module):
    def __init__(self, input_dims, static_dim, hidden_size=1024, n_layers=2):
        super(LSTM_NetworkBase, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size,
                            batch_first=True, num_layers=n_layers, dropout=0.2)
        self.self_attention = SelfAttention(hidden_size + static_dim)

    def forward(self, state, static_input):
        lstm_output, _ = self.lstm(state)
        if lstm_output.dim() == 2:
            lstm_output = lstm_output.unsqueeze(0)

        # Process static input
        batch_size, seq_len, _ = lstm_output.shape
        static_input = static_input.unsqueeze(-1)
        static_input_expanded = static_input.expand(batch_size, seq_len, -1)

        # Combine LSTM output and static input
        combined_input = torch.cat((lstm_output, static_input_expanded), dim=2)
        attention_output, _ = self.self_attention(combined_input)

        return attention_output

class LSTM_ActorNetwork(LSTM_NetworkBase):
    def __init__(self, n_actions, input_dims, static_dim, hidden_size=1024):
        super(LSTM_ActorNetwork, self).__init__(input_dims, static_dim, hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 1024),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(1/16)
        )
        self.policy = nn.Linear(256, n_actions)

    def forward(self, state, static_input):
        attention_output = super().forward(state, static_input).squeeze(0)
        x = self.fc1(attention_output)
        action_probs = torch.softmax(self.policy(x), dim=-1)
        return action_probs

class LSTM_CriticNetwork(LSTM_NetworkBase):
    def __init__(self, input_dims, static_dim, hidden_size=1024):
        super(LSTM_CriticNetwork, self).__init__(input_dims, static_dim, hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 1024),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(1/16)
        )
        self.value = nn.Linear(256, 1)

    def forward(self, state, static_input):
        attention_output = super().forward(state, static_input).squeeze(0)
        x = self.fc1(attention_output)
        value = self.value(x)
        return value


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.1, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO repair cuda
        # self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.entropy_coefficient = entropy_coefficient
        self.l1_lambda = l1_lambda
        self.static_dim = 1

        # Initialize the actor and critic networks
        self.actor = LSTM_ActorNetwork(n_actions, input_dims, self.static_dim).to(self.device)
        self.critic = LSTM_CriticNetwork(input_dims, self.static_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done, static_input):
        self.memory.store_memory(state, action, probs, vals, reward, done, static_input)

    def learn(self):
        for _ in range(self.n_epochs):
            # Generating the data for the entire batch
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, static_input_arr, batches = self.memory.generate_batches()

            values = torch.tensor(vals_arr, dtype=torch.float).to(self.device)
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculating Advantage
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage, dtype=torch.float).to(self.device)

            # Creating mini-batches
            num_samples = len(state_arr)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.mini_batch_size):
                # Extract indices for the mini-batch
                minibatch_indices = indices[start_idx:start_idx + self.mini_batch_size]
                static_input_batch = static_input_arr[minibatch_indices]

                # Extract data for the current mini-batch
                batch_states = torch.tensor(state_arr[minibatch_indices], dtype=torch.float).to(self.device)
                batch_actions = torch.tensor(action_arr[minibatch_indices], dtype=torch.long).to(self.device)
                batch_old_probs = torch.tensor(old_prob_arr[minibatch_indices], dtype=torch.float).to(self.device)
                batch_advantages = advantage[minibatch_indices]
                batch_values = values[minibatch_indices]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Actor Network Loss with Entropy Regularization
                probs = self.actor(batch_states, torch.tensor(static_input_batch, dtype=torch.float).to(self.device))
                dist = torch.distributions.Categorical(probs)
                new_probs = dist.log_prob(batch_actions)
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                weighted_probs = batch_advantages * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                weighted_clipped_probs = clipped_probs * batch_advantages
                entropy = dist.entropy().mean()  # Entropy of the policy distribution
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - self.entropy_coefficient * entropy
                l1_loss_actor = sum(torch.sum(torch.abs(param)) for param in self.actor.parameters())
                actor_loss += self.l1_lambda * l1_loss_actor

                # Critic Network Loss
                critic_value = self.critic(batch_states, torch.tensor(static_input_batch, dtype=torch.float).to(self.device)).squeeze()
                returns = batch_advantages + batch_values
                critic_loss = nn.functional.mse_loss(critic_value, returns)
                l1_loss_critic = sum(torch.sum(torch.abs(param)) for param in self.critic.parameters())
                critic_loss += self.l1_lambda * l1_loss_critic

                # Gradient Calculation and Optimization Step
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Clear memory
        self.memory.clear_memory()

    def choose_action(self, observation, static_input):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        static_input = torch.tensor([static_input], dtype=torch.float).to(self.device)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        probs = self.actor(state, static_input)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state, static_input)

        return action.item(), log_prob.item(), value.item()

    def get_action_probabilities(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        static_input = torch.tensor([0], dtype=torch.float).to(self.device)

        # Get the action probabilities from the actor network
        action_probs = self.actor(state, static_input).cpu().detach().numpy()
        return action_probs

    def choose_best_action(self, observation):
        # Get action probabilities
        action_probs = self.get_action_probabilities(observation)
        # Choose the action with the highest probability
        best_action = np.argmax(action_probs)
        return best_action


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000, leverage=1, window_size_statics=30):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables

        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage

        # for reward function
        self.holding_duration = 0
        self.rolling_returns = []
        self.rolling_downside_returns = []
        self.window_size_statics = window_size_statics

        # Define action and observation space
        self.action_space = spaces.Discrete(3)

        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        return input_dims

    def reset(self, observation=None):
        if observation is not None:
            self.current_step = observation + self.look_back
        else:
            self.current_step = self.look_back

        self.balance = self.initial_balance
        self.done = False
        return self._next_observation()

    def _create_base_observation(self):
        start = max(self.current_step - self.look_back, 0)
        end = self.current_step
        return self.df[self.variables].iloc[start:end].values

    def _next_observation(self):
        start = max(self.current_step - self.look_back, 0)
        end = self.current_step

        # Apply scaling based on 'edit' property of each variable
        scaled_observation = []
        for variable in self.variables:
            data = self.df[variable['variable']].iloc[start:end].values
            if variable['edit'] == 'standardize':
                scaled_data = standardize_data(data)
            elif variable['edit'] == 'normalize':
                scaled_data = normalize_data(data)
            else:  # Default to normalization
                scaled_data = data

            scaled_observation.extend(scaled_data)
        return np.array(scaled_observation)

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0):
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns)

    def calculate_sortino_ratio(self, returns, risk_free_rate=0):
        excess_returns = returns - risk_free_rate
        downside_returns = [r for r in excess_returns if r < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(downside_returns)

    def step(self, action):
        # Existing action mapping
        action_mapping = {0: -1, 1: 0, 2: 1}
        mapped_action = action_mapping[action]

        # Get current and next price
        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step + 1]

        # Calculate log return based on action
        log_return = math.log(next_price / current_price) if current_price != 0 else 0
        reward = 0

        if mapped_action == 1:  # Buying
            reward = log_return
        elif mapped_action == -1:  # Selling
            reward = -log_return

        # Apply leverage to the base reward
        reward = reward * self.leverage

        # Calculate cost based on action and current position
        if mapped_action != self.current_position:
            if mapped_action == 0:
                provision = 0
            else:
                # double the provision as it is applied on open and close of position (in this approach we take provision for both open and close on opening) in order to agent have higher rewards for deciding to 0 position
                provision = math.log(1 - 2 * self.provision)
        else:
            provision = 0

        reward += provision

        # Update rolling returns
        self.rolling_returns.append(reward)

        # Update the balance
        self.balance *= math.exp(reward)  # Update balance using exponential of reward before applying other penalties

        # Update current position and step
        self.current_position = mapped_action
        self.current_step += 1

        if len(self.rolling_returns) >= self.window_size_statics:
            sharpe_ratio = self.calculate_sharpe_ratio(np.array(self.rolling_returns[-self.window_size_statics:]))
            sortino_ratio = self.calculate_sortino_ratio(np.array(self.rolling_downside_returns[-self.window_size_statics:]))
        else:
            sharpe_ratio = sortino_ratio = 0

        # Update holding duration
        if mapped_action == self.current_position and mapped_action != 0:
            self.holding_duration += 1
        else:
            self.holding_duration = 0  # Reset if the position changes or goes to 0

        # Calculate holding penalty
        if self.holding_duration > 10:
            holding_penalty = -0.0001 * self.holding_duration

        else:
            holding_penalty = 0
        # calculate reward with other penalties
        final_reward = reward + holding_penalty  #  + sharpe_ratio/100 + sortino_ratio/100
        # Check if the episode is done
        if self.current_step >= len(self.df) - 1:
            self.done = True
        return self._next_observation(), final_reward, self.done, {}

# Example usage
# Stock market variables
df = load_data(['EURUSD', 'USDJPY', 'EURJPY'], '4H')

indicators = [
    {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
    {"indicator": "ATR", "mkf": "EURUSD", "length": 24},]

add_indicators(df, indicators)
returns = [
    {"mkf": "EURUSD", "price_type": "Close", "n": 1},
    {"mkf": 'USDJPY', "price_type": "Close", "n": 1},
    {"mkf": 'EURJPY', "price_type": "Close", "n": 1},
]

for return_info in returns:
    mkf = return_info["mkf"]
    price_type = return_info.get("price_type", "Close")
    n = return_info.get("n", 1)
    df = add_returns(df, mkf, price_type, n)
    df = add_log_returns(df, mkf, price_type, n)

df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")]/100
df = df.dropna()
start_date = '2013-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df[("Log_Returns_Close_1", "EURUSD")] = df[("Log_Returns_Close_1", "EURUSD")]*25
df[("Log_Returns_Close_1", "USDJPY")] = df[("Log_Returns_Close_1", "USDJPY")]*25
df[("Log_Returns_Close_1", "EURJPY")] = df[("Log_Returns_Close_1", "EURJPY")]*25
df_train = df[start_date:validation_date]
df_validation = df[validation_date:test_date]
df_test = df[test_date:]
variables = [
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "EURJPY"), "edit": "normalize"},
    {"variable": ("RSI_14", "EURUSD"), "edit": "None"},
    {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
    {"variable": ("Log_Returns_Close_1", "EURUSD"), "edit": "None"},
    {"variable": ("Log_Returns_Close_1", "USDJPY"), "edit": "None"},
    {"variable": ("Log_Returns_Close_1", "EURJPY"), "edit": "None"},
]
tradable_markets = 'EURUSD'
window_size = '1Y'
starting_balance = 10000
look_back = 18
provision = 0.0001  # 0.001, cant be too high as it would not learn to trade

# Training parameters
batch_size = 2048
epochs = 30  # 40
mini_batch_size = 128
leverage = 1
weight_decay = 0.0005
l1_lambda = 1e-5
# This is a transformer model with self-attention for time series forecasting
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

agent = PPO_Agent(n_actions=env.action_space.n,
                  input_dims=env.calculate_input_dims(),
                  gamma=0.5,
                  alpha=0.02,
                  gae_lambda=0.8,
                  policy_clip=0.2,
                  entropy_coefficient=0.1,  # maybe try higher entropy coefficient
                  batch_size=batch_size,
                  n_epochs=epochs,
                  mini_batch_size=mini_batch_size,
                  weight_decay=weight_decay,
                  l1_lambda=l1_lambda)

num_episodes = 100  # 250

total_rewards = []
episode_durations = []
total_balances = []

# Assuming df_train is your DataFrame
rolling_datasets = rolling_window_datasets(df_train, window_size=window_size,  look_back=look_back)

# Create a DataFrame to store the backtest results
index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
columns = ['Final Balance', 'Dataset Index']  # Add 'Dataset Index' column
backtest_results = pd.DataFrame(index=index, columns=columns)

# Use 'cycle' to endlessly iterate over the rolling_datasets
dataset_iterator = cycle(rolling_datasets)

for episode in tqdm(range(num_episodes)):
    # Randomly select one dataset from the rolling datasets
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
        current_position = env.current_position
        action, prob, val = agent.choose_action(observation, current_position)
        static_input = current_position
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, prob, val, reward, done, static_input)
        observation = observation_
        cumulative_reward += reward

        # Check if enough data is collected or if the dataset ends
        if len(agent.memory.states) >= agent.memory.batch_size or done:
            agent.learn()
            agent.memory.clear_memory()

    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(cumulative_reward)
    episode_durations.append(episode_time)
    total_balances.append(env.balance)

    # Backtesting
    validation_balance, validation_number_of_trades = generate_predictions_and_backtest(df_validation, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage)
    test_balance, test_number_of_trades = generate_predictions_and_backtest(df_test, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage)
    backtest_results.loc[(episode, 'validation'), 'Final Balance'] = validation_balance
    backtest_results.loc[(episode, 'test'), 'Final Balance'] = test_balance
    backtest_results.loc[(episode, 'validation'), 'Number of Trades'] = validation_number_of_trades
    backtest_results.loc[(episode, 'test'), 'Number of Trades'] = test_number_of_trades
    backtest_results.loc[(episode, 'validation'), 'Dataset Index'] = dataset_index
    backtest_results.loc[(episode, 'test'), 'Dataset Index'] = dataset_index

    print(f"Completed learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
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

# Plotting the results after all episodes
plt.plot(episode_durations, color='red')
plt.title('Episode Duration Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Plotting the results after all episodes
plt.plot(total_balances, color='green')
plt.title('Total Balance Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Extracting data for plotting
validation_pnl = backtest_results.loc[(slice(None), 'validation'), 'Final Balance']
test_pnl = backtest_results.loc[(slice(None), 'test'), 'Final Balance']

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), validation_pnl.values, label='Validation Total PnL', marker='o')
plt.plot(range(num_episodes), test_pnl.values, label='Test Total PnL', marker='x')

plt.title('Total PnL Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total PnL')
plt.legend()
plt.show()


# final prediction agent
# Validation
df_validation_probs = df_validation.copy()
predictions_df = pd.DataFrame(index=df_validation.index, columns=['Predicted_Action'])
validation_env = Trading_Environment_Basic(df_validation, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

for validation_observation in range(len(df_validation) - validation_env.look_back):
    observation = validation_env.reset(validation_observation)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[validation_observation + validation_env.look_back] = action

# Merge with df_validation
df_validation_with_predictions = df_validation.copy()
df_validation_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1

# final prediction with probabilities
validation_env_probs = Trading_Environment_Basic(df_validation_probs, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
action_probabilities = []

for validation_observation in range(len(df_validation_probs) - validation_env_probs.look_back):
    observation = validation_env_probs.reset(validation_observation)  # Reset environment to the specific observation
    probs = agent.get_action_probabilities(observation)
    action_probabilities.append(probs[0])

# Convert the list of probabilities to a DataFrame
probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Do_nothing', 'Long'])

# Join with the original validation DataFrame
df_validation_with_probabilities = df_validation_probs.iloc[validation_env_probs.look_back:].reset_index(drop=True)
df_validation_with_probabilities = pd.concat([df_validation_with_probabilities, probabilities_df], axis=1)

# TEST
df_test_probs = df_test.copy()
predictions_df = pd.DataFrame(index=df_test.index, columns=['Predicted_Action'])
test_env = Trading_Environment_Basic(df_test, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

for test_observation in range(len(df_test) - test_env.look_back):
    observation = test_env.reset(test_observation)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[test_observation + test_env.look_back] = action

# Merge with df_test
df_test_with_predictions = df_test.copy()
df_test_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1


# final prediction with probabilities
test_env_probs = Trading_Environment_Basic(df_test_probs, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
action_probabilities = []

for test_observation in range(len(df_test_probs) - test_env_probs.look_back):
    observation = test_env_probs.reset(test_observation)  # Reset environment to the specific observation
    probs = agent.get_action_probabilities(observation)
    action_probabilities.append(probs[0])

# Convert the list of probabilities to a DataFrame
probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Do_nothing', 'Long'])

# Join with the original test DataFrame
df_test_with_probabilities = df_test_probs.iloc[test_env_probs.look_back:].reset_index(drop=True)
df_test_with_probabilities = pd.concat([df_test_with_probabilities, probabilities_df], axis=1)

# TRAIN
df_train_probs = df_train.copy()
predictions_df = pd.DataFrame(index=df_train.index, columns=['Predicted_Action'])
train_env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

for train_observation in range(len(df_train) - train_env.look_back):
    observation = train_env.reset(train_observation)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[train_observation + train_env.look_back] = action

# Merge with df_train
df_train_with_predictions = df_train.copy()
df_train_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1


train_env_probs = Trading_Environment_Basic(df_train_probs, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
action_probabilities = []

for train_observation in range(len(df_train_probs) - train_env_probs.look_back):
    observation = train_env_probs.reset(train_observation)  # Reset environment to the specific observation
    probs = agent.get_action_probabilities(observation)
    action_probabilities.append(probs[0])

# Convert the list of probabilities to a DataFrame
probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Do_nothing', 'Long'])

# Join with the original train DataFrame
df_train_with_probabilities = df_train_probs.iloc[train_env_probs.look_back:].reset_index(drop=True)
df_train_with_probabilities = pd.concat([df_train_with_probabilities, probabilities_df], axis=1)
