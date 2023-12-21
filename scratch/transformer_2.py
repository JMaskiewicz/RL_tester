import torch.nn.functional as F

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

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators
from data.edit import normalize_data, standardize_data

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# TODO use deque instead of list
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(float(reward))
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


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
        weights = F.softmax(energy, dim=1)
        outputs = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class TFT_ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, hidden_size=128):
        super(TFT_ActorNetwork, self).__init__()
        self.self_attention = SelfAttention(hidden_size)

        # Assume lstm_output size to be hidden_size
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size, batch_first=True)

        # Final layer to output action probabilities
        self.policy = nn.Linear(hidden_size, n_actions)

    def forward(self, state):
        # Assuming state shape is [batch, sequence, features]
        lstm_output, (hidden, cell) = self.lstm(state)
        attention_output, weights = self.self_attention(lstm_output)

        action_probs = torch.softmax(self.policy(attention_output), dim=-1)
        return action_probs


class TFT_CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_size=128):
        super(TFT_CriticNetwork, self).__init__()
        self.self_attention = SelfAttention(hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size, batch_first=True)

        # Final layer to output value estimate
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        # Assuming state shape is [batch, sequence, features]
        lstm_output, (hidden, cell) = self.lstm(state)
        attention_output, weights = self.self_attention(lstm_output)

        value = self.value(attention_output)
        return value


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5):
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

        # Initialize the actor and critic networks
        self.actor = TFT_ActorNetwork(n_actions, input_dims).to(self.device)
        self.critic = TFT_CriticNetwork(input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self):
        for _ in range(self.n_epochs):
            # Generating the data for the entire batch
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, _ = self.memory.generate_batches()

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

                # Extract data for the current mini-batch
                batch_states = torch.tensor(state_arr[minibatch_indices], dtype=torch.float).to(self.device)
                batch_actions = torch.tensor(action_arr[minibatch_indices], dtype=torch.long).to(self.device)
                batch_old_probs = torch.tensor(old_prob_arr[minibatch_indices], dtype=torch.float).to(self.device)
                batch_advantages = advantage[minibatch_indices]
                batch_values = values[minibatch_indices]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Actor Network Loss with Entropy Regularization
                probs = self.actor(batch_states)
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
                critic_value = self.critic(batch_states).squeeze()
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

    def choose_action(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        probs = self.actor(state)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        return action.item(), log_prob.item(), value.item()

    def get_action_probabilities(self, observation):
        pass

    def choose_best_action(self, observation):
        pass


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, current_positions=True, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000, leverage=1):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables
        self.current_positions = current_positions
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        if self.current_positions:
            self.observation_space = spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(look_back + 1,),  # +1 for current position
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(look_back,),
                                                dtype=np.float32)

        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        if self.current_positions:
            input_dims += 1  # Add one more dimension for current position
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

        if self.current_positions:
            scaled_observation = np.append(scaled_observation, (self.current_position+1)/2)

        return np.array(scaled_observation)

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
                provision = math.log(1 - 2 * self.provision)  # double the provision as it is applied on open and close of position (in this approach we take provision for both open and close on opening) in order to agent have higher rewards for deciding to 0 position
        else:
            provision = 0
        reward += provision

        # Update the balance
        self.balance *= math.exp(reward)  # Update balance using exponential of reward

        # Update current position and step
        self.current_position = mapped_action
        self.current_step += 1

        # Check if the episode is done
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._next_observation(), reward, self.done, {}

# Example usage
# Stock market variables
df = load_data(['EURUSD', 'USDJPY'], '1D')

indicators = [
    {"indicator": "RSI", "mkf": "EURUSD", "length": 14},]
add_indicators(df, indicators)

df = df.dropna()
start_date = '2020-10-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[start_date:validation_date]
df_validation = df[validation_date:test_date]
df_test = df[test_date:]
variables = [
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("RSI_14", "EURUSD"), "edit": "normalize"},
]
tradable_markets = 'EURUSD'
window_size = '6M'
starting_balance = 10000
look_back = 20
provision = 0.001  # 0.001, cant be too high as it would not learn to trade

# Training parameters
batch_size = 2048
epochs = 5  # 40
mini_batch_size = 256
leverage = 30
weight_decay = 0.0005
l1_lambda = 1e-5
# This is a transformer model with self-attention for time series forecasting
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, current_positions=False, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)


agent = PPO_Agent(n_actions=env.action_space.n,
                  input_dims=env.calculate_input_dims(),
                  gamma=0.7,
                  alpha=0.005,
                  gae_lambda=0.8,
                  policy_clip=0.1,
                  entropy_coefficient=0.01,
                  batch_size=batch_size,
                  n_epochs=epochs,
                  mini_batch_size=mini_batch_size,
                  weight_decay=weight_decay,
                  l1_lambda=l1_lambda)

num_episodes = 10  # 100

total_rewards = []
episode_durations = []
total_balances = []

# Assuming df_train is your DataFrame
rolling_datasets = rolling_window_datasets(df_train, window_size=window_size,  look_back=look_back)

for episode in tqdm(range(num_episodes)):
    # Randomly select one dataset from the rolling datasets
    window_df = random.choice(rolling_datasets)

    print(f"\nEpisode {episode + 1}: Learning from dataset with Start Date = {window_df.index.min()}, End Date = {window_df.index.max()}, len = {len(window_df)}")
    # Create a new environment with the randomly selected window's data
    env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables, current_positions=False, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    observation = env.reset()
    done = False
    cumulative_reward = 0
    start_time = time.time()
    initial_balance = env.balance

    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, prob, val, reward, done)
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

    print(f"Completed learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
    print("-----------------------------------")

