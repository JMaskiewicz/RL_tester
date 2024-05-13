"""
Transformer based PPO agent for trading version 4.2

# this code
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.

# TODO LIST
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- HRL (Hierarchical Reinforcement Learning): Use hierarchical reinforcement learning to learn sub-policies for different tasks. Master agent would distribute capital among sub-agents for different tickers.

Some notes on the code:
- learning of the agent is fast (3.38s for batch of 8192 and mini-batch of 256)
- higher number of epochs agent would less likely to take a neutral position

Reward testing:tral p
- higher penalty for wrong actions this would make agent more likely to take a neuosition
- higher number of epochs agent would less likely to take a neutral position
- premium for holding position agent would less likely to change position
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ExponentialLR
import time
from numba import jit
import math

from data.function.load_data import load_data_parallel
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
from functions.utilis import save_model
import backtest.backtest_functions.functions as BF
from functions.utilis import prepare_backtest_results, generate_index_labels, get_time

# import environment class
from trading_environment.environment import Trading_Environment_Basic

# import benchmark agents
from backtest.benchmark_agents import Buy_and_hold_Agent, Sell_and_hold_Agent
"""
Reward Calculation function is the most crucial part of the RL algorithm. It is the function that determines the reward the agent receives for its actions.
"""

@jit(nopython=True)
def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
    # Calculate the normal return
    if previous_close != 0:
        normal_return = (current_close - previous_close) / previous_close
    else:
        normal_return = 0
    '''
    magnitude_of_reward = abs(normal_return) ** (1 / 2) * 100

    # Apply the sign of normal_return to the magnitude of the reward
    if normal_return < 0:
        reward = -magnitude_of_reward * current_position
    else:
        reward = magnitude_of_reward * current_position
    '''

    # Calculate the base reward
    reward = normal_return * current_position * 100

    # Penalize the agent for taking the wrong action
    if reward < 0:
        reward *= 1.1  # penalty for wrong action

    # Calculate the cost of provision if the position has changed, and it's not neutral (0).
    if current_position != previous_position and abs(current_position) == 1:
        provision_cost = - provision * 100  # penalty for changing position
    elif current_position == previous_position and abs(current_position) == 1:
        provision_cost = + provision * 0
    else:
        provision_cost = 0

    # Apply the provision cost
    reward += provision_cost

    # Scale the reward to enhance its significance for the learning process
    final_reward = reward

    return final_reward

class PPOMemory:
    def __init__(self, batch_size, device):
        self.states = None
        self.probs = None
        self.actions = None
        self.vals = None
        self.rewards = None
        self.dones = None
        self.static_states = None
        self.batch_size = batch_size

        self.clear_memory()
        self.device = device

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, self.static_states, batches

    def store_memory(self, state, action, probs, vals, reward, done, static_state):
        self.states.append(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        self.actions.append(torch.tensor(action, dtype=torch.long).unsqueeze(0))
        self.probs.append(torch.tensor(probs, dtype=torch.float).unsqueeze(0))
        self.vals.append(torch.tensor(vals, dtype=torch.float).unsqueeze(0))
        self.rewards.append(torch.tensor(reward, dtype=torch.float).unsqueeze(0))
        self.dones.append(torch.tensor(done, dtype=torch.bool).unsqueeze(0))
        self.static_states.append(torch.tensor(static_state, dtype=torch.float).unsqueeze(0))

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.static_states = []

    def stack_tensors(self):
        self.states = torch.cat(self.states, dim=0).to(self.device)
        self.actions = torch.cat(self.actions, dim=0).to(self.device)
        self.probs = torch.cat(self.probs, dim=0).to(self.device)
        self.vals = torch.cat(self.vals, dim=0).to(self.device)
        self.rewards = torch.cat(self.rewards, dim=0).to(self.device)
        self.dones = torch.cat(self.dones, dim=0).to(self.device)
        self.static_states = torch.cat(self.static_states, dim=0).to(self.device)


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, n_heads=4, n_layers=2, dropout_rate=1/4, static_input_dims=1):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.static_input_dims = static_input_dims
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        encoder_layers = TransformerEncoderLayer(d_model=input_dims, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)

        self.max_position_embeddings = 128
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.max_position_embeddings, input_dims))
        self.fc_static = nn.Linear(static_input_dims, input_dims)

        self.fc1 = nn.Linear(input_dims * 2, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, n_actions)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dynamic_state, static_state):
        batch_size, seq_length, _ = dynamic_state.size()
        positional_encoding = self.positional_encoding[:, :seq_length, :].expand(batch_size, -1, -1)

        dynamic_state = dynamic_state + positional_encoding
        transformer_out = self.transformer_encoder(dynamic_state)

        static_state_encoded = self.fc_static(static_state.unsqueeze(1))
        combined_features = torch.cat((transformer_out[:, -1, :], static_state_encoded.squeeze(1)), dim=1)

        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return self.softmax(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_heads=4, n_layers=2, dropout_rate=1 / 4, static_input_dims=1):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.static_input_dims = static_input_dims
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        encoder_layers = TransformerEncoderLayer(d_model=input_dims, nhead=n_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)

        self.max_position_embeddings = 128
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.max_position_embeddings, input_dims))
        self.fc_static = nn.Linear(static_input_dims, input_dims)

        self.fc1 = nn.Linear(input_dims * 2, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, dynamic_state, static_state):
        batch_size, seq_length, _ = dynamic_state.size()
        positional_encoding = self.positional_encoding[:, :seq_length, :].expand(batch_size, -1, -1)

        dynamic_state = dynamic_state + positional_encoding
        transformer_out = self.transformer_encoder(dynamic_state)

        static_state_encoded = self.fc_static(static_state.unsqueeze(1))
        combined_features = torch.cat((transformer_out[:, -1, :], static_state_encoded.squeeze(1)), dim=1)

        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Transformer_PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024,
                 n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, ec_decay_rate=0.999, weight_decay=0.0001, l1_lambda=1e-5,
                 static_input_dims=1, lr_decay_rate=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Not sure why CPU is faster
        # self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.gamma = gamma  # Discount factor
        self.policy_clip = policy_clip  # PPO policy clipping parameter
        self.n_epochs = n_epochs  # Number of optimization epochs per batch
        self.gae_lambda = gae_lambda  # Generalized Advantage Estimation lambda
        self.mini_batch_size = mini_batch_size  # Size of mini-batches for optimization
        self.entropy_coefficient = entropy_coefficient  # Entropy coefficient for encouraging exploration
        self.ec_decay_rate = ec_decay_rate  # Entropy coefficient decay rate
        self.l1_lambda = l1_lambda  # L1 regularization coefficient
        self.lr_decay_rate = lr_decay_rate  # Learning rate decay rate
        self.n_actions = n_actions  # Number of actions

        # Initialize the actor and critic networks with static input dimensions
        self.actor = ActorNetwork(self.n_actions, input_dims, static_input_dims=static_input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims, static_input_dims=static_input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        # Learning rate schedulers
        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=self.lr_decay_rate)
        self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=self.lr_decay_rate)

        # Memory for storing experiences
        self.memory = PPOMemory(batch_size, self.device)

        # track the generation of the agent
        self.generation = 0

    def store_transition(self, state, action, probs, vals, reward, done, static_state):
        # Include static_state in the memory storage
        self.memory.store_memory(state, action, probs, vals, reward, done, static_state)

    def learn(self):
        # track the time it takes to learn
        start_time = time.time()
        print('\n', "-" * 100)
        # Set the actor and critic networks to training mode
        self.actor.train()
        self.critic.train()

        # Stack the tensors in the memory
        self.memory.stack_tensors()

        # Generating the data for the entire batch, including static states
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, static_states_arr, batches = self.memory.generate_batches()

        # Convert arrays to tensors and move to the device
        state_arr = state_arr.clone().detach().to(self.device)  # Dynamic states ie time series data
        action_arr = action_arr.clone().detach().to(self.device)  # Actions
        old_prob_arr = old_prob_arr.clone().detach().to(self.device)  # Old action probabilities
        vals_arr = vals_arr.clone().detach().to(self.device)  # State values
        reward_arr = reward_arr.clone().detach().to(self.device)  # Rewards
        dones_arr = dones_arr.clone().detach().to(self.device)  # Done flags
        static_states_arr = static_states_arr.clone().detach().to(self.device)  # Static states

        # Compute advantages and discounted rewards
        advantages, discounted_rewards = self.compute_discounted_rewards(reward_arr, vals_arr,
                                                                         dones_arr)
        advantages = advantages.clone().detach().to(self.device)
        discounted_rewards = discounted_rewards.clone().detach().to(self.device)

        # Loop through the optimization epochs
        for _ in range(self.n_epochs):

            # Creating mini-batches and training
            num_samples = len(state_arr)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            # Loop through mini-batches
            for start_idx in range(0, num_samples, self.mini_batch_size):
                minibatch_indices = indices[start_idx:start_idx + self.mini_batch_size]

                # Convert arrays to tensors and move to the device
                batch_states = state_arr[minibatch_indices].clone().detach().to(self.device)
                batch_actions = action_arr[minibatch_indices].clone().detach().to(self.device)
                batch_old_probs = old_prob_arr[minibatch_indices].clone().detach().to(self.device)
                batch_advantages = advantages[minibatch_indices].clone().detach().to(self.device)
                batch_returns = discounted_rewards[minibatch_indices].clone().detach().to(self.device)
                batch_static_states = static_states_arr[minibatch_indices].clone().detach().to(self.device)

                # Zero the gradients before the backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Calculate actor and critic losses, include static states in forward passes
                new_probs, dist_entropy, actor_loss, critic_loss = self.calculate_loss(batch_states,
                                                                                       batch_actions,
                                                                                       batch_old_probs,
                                                                                       batch_advantages,
                                                                                       batch_returns,
                                                                                       batch_static_states)

                # Perform backpropagation and optimization steps for both actor and critic networks
                actor_loss.backward()
                self.actor_optimizer.step()
                critic_loss.backward()
                self.critic_optimizer.step()

            # Decay learning rate
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        # Clear memory after learning
        self.memory.clear_memory()

        # Increment generation of the agent
        self.generation += 1

        # decay entropy coefficient
        self.entropy_coefficient *= self.ec_decay_rate

        # track the time it takes to learn
        end_time = time.time()
        episode_time = end_time - start_time

        # print the time it takes to learn
        print(f"Learning of agent generation {self.generation} completed in {episode_time} seconds")
        print("-" * 100)

    def calculate_loss(self, batch_states, batch_actions, batch_old_probs, batch_advantages,
                       batch_returns, batch_static_states):
        # Ensure batch_states has the correct 3D shape: [batch size, sequence length, feature dimension]
        if batch_states.dim() == 2:
            batch_states = batch_states.unsqueeze(1)

        # Actor loss calculations
        new_probs = self.actor(batch_states, batch_static_states)

        # Calculate the probability ratio and the surrogate loss
        dist = torch.distributions.Categorical(new_probs)

        # Calculate the log probability of the action in the distribution
        new_log_probs = dist.log_prob(batch_actions)

        # Calculate the probability ratio
        prob_ratios = torch.exp(new_log_probs - batch_old_probs)

        # Calculate the surrogate loss
        surr1 = prob_ratios * batch_advantages

        # Clipped surrogate loss
        surr2 = torch.clamp(prob_ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages

        # Actor loss
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * dist.entropy().mean()

        # Critic loss calculations
        critic_values = self.critic(batch_states, batch_static_states).squeeze(-1)
        critic_loss = (batch_returns - critic_values).pow(2).mean()

        return new_probs, dist.entropy(), actor_loss, critic_loss

    def compute_discounted_rewards(self, rewards, values, dones):
        n = len(rewards)
        advantages = torch.zeros_like(rewards)
        discounted_rewards = torch.zeros_like(rewards)

        # Ensure 'dones' is a float tensor for arithmetic operations
        dones_float = dones.float()

        # Append zero to the end of the values tensor for the terminal value calculation
        next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])

        # Compute deltas using tensors; all operations should now be valid
        deltas = rewards + self.gamma * next_values * (1 - dones_float) - values

        lastgaelam = 0
        for t in reversed(range(n)):
            lastgaelam = deltas[t] + self.gamma * self.gae_lambda * lastgaelam * (1 - dones_float[t])
            advantages[t] = lastgaelam

        discounted_rewards = advantages + values

        return advantages, discounted_rewards

    @torch.no_grad()
    def choose_action(self, observation, static_input):
        # Ensure observation is a NumPy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # Reshape observation to [1, sequence length, feature dimension]
        observation = observation.reshape(1, -1, observation.shape[-1])

        # Convert observation and static_input to tensors and move them to the appropriate device
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        static_input_tensor = torch.tensor([static_input], dtype=torch.float).to(self.device)

        # Ensure state has the correct 3D shape: [batch size, sequence length, feature dimension]
        if state.dim() != 3:  # Add missing dimensions if necessary
            state = state.view(1, -1, state.size(-1))

        # Pass the state and static_input_tensor through the actor network to get the action probabilities
        probs = self.actor(state, static_input_tensor)

        # Create a categorical distribution over the list of probabilities of actions
        dist = torch.distributions.Categorical(probs)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability of the action in the distribution
        log_prob = dist.log_prob(action)

        # Pass the state and static_input_tensor through the critic network to get the state value
        value = self.critic(state, static_input_tensor)

        # Return the sampled action, its log probability, and the state value
        # Convert tensors to Python numbers using .item()
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_action_probabilities(self, observation, static_input):
        # Ensure observation is a NumPy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # Reshape observation to [1, sequence length, feature dimension]
        observation = observation.reshape(1, -1, observation.shape[-1])

        # Convert observation and static_input to tensors and move them to the appropriate device
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        static_input_tensor = torch.tensor([static_input], dtype=torch.float).to(self.device)

        # Ensure state has the correct 3D shape: [batch size, sequence length, feature dimension]
        if state.dim() != 3:  # Add missing dimensions if necessary
            state = state.view(1, -1, state.size(-1))

        # Pass the state and static_input_tensor through the actor network to get the action probabilities
        action_probs = self.actor(state, static_input_tensor)

        # Ensure action_probs does not contain any gradients and convert it to a NumPy array
        action_probs = action_probs.detach().cpu().numpy()

        # Squeeze the batch dimension from action_probs since we're dealing with a single observation
        action_probs = np.squeeze(action_probs, axis=0)

        # Return the action probabilities as a NumPy array
        return action_probs

    @torch.no_grad()
    def choose_best_action(self, observation, static_input):
        # Use the get_action_probabilities method to get the action probabilities for the given observation and static input
        action_probs = self.get_action_probabilities(observation, static_input)

        # Choose the action with the highest probability
        best_action = np.argmax(action_probs)

        return best_action

    def get_name(self):
        """
        Returns the class name of the instance.
        """
        return self.__class__.__name__


if __name__ == '__main__':
    # time the execution
    start_time_X = time.time()
    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Example usage
    # Stock market variables
    df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD'], '1D')

    indicators = [
        {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
        {"indicator": "ATR", "mkf": "EURUSD", "length": 36},
        {"indicator": "MACD", "mkf": "EURUSD"},
        {"indicator": "Stochastic", "mkf": "EURUSD"}, ]

    return_indicators = [
        {"price_type": "Close", "mkf": "EURUSD"},
        {"price_type": "Close", "mkf": "USDJPY"},
        {"price_type": "Close", "mkf": "EURJPY"},
        {"price_type": "Close", "mkf": "GBPUSD"},
    ]
    add_indicators(df, indicators)
    add_returns(df, return_indicators)

    add_time_sine_cosine(df, '1W')
    df[("sin_time_1W", "")] = df[("sin_time_1W", "")] / 2 + 0.5
    df[("cos_time_1W", "")] = df[("cos_time_1W", "")] / 2 + 0.5
    df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")] / 100

    df = df.dropna()
    # data before 2006 has some missing values ie gaps in the data, also in march, april 2023 there are some gaps
    start_date = '2005-01-01'  # worth to keep 2008 as it was a financial crisis
    validation_date = '2017-12-31'
    test_date = '2019-01-01'
    df_train, df_validation, df_test = df[start_date:validation_date], df[validation_date:test_date], df[test_date:'2023-01-01']

    variables = [
        {"variable": ("Close", "USDJPY"), "edit": "normalize"},
        {"variable": ("Close", "EURUSD"), "edit": "normalize"},
        {"variable": ("Close", "EURJPY"), "edit": "normalize"},
        {"variable": ("Close", "GBPUSD"), "edit": "normalize"},
        {"variable": ("RSI_14", "EURUSD"), "edit": "standardize"},
        {"variable": ("ATR_36", "EURUSD"), "edit": "standardize"},
        {"variable": ("K%", "EURUSD"), "edit": "standardize"},
        {"variable": ("D%", "EURUSD"), "edit": "standardize"},
        {"variable": ("MACD_Line", "EURUSD"), "edit": "standardize"},
        {"variable": ("Signal_Line", "EURUSD"), "edit": "standardize"},
        {"variable": ("cos_time_1W", ""), "edit": None},
        {"variable": ("Returns_Close", "EURUSD"), "edit": None},
        {"variable": ("Returns_Close", "USDJPY"), "edit": None},
        {"variable": ("Returns_Close", "EURJPY"), "edit": None},
        {"variable": ("Returns_Close", "GBPUSD"), "edit": None},
    ]

    tradable_markets = 'EURUSD'
    window_size = '1Y'
    starting_balance = 10000
    look_back = 20
    # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001
    provision = 0.0001  # 0.001, cant be too high as it would not learn to trade

    # Training parameters
    leverage = 1  # 30
    num_episodes = 5000

    # Create an instance of the agent
    agent = Transformer_PPO_Agent(n_actions=3,  # sell, hold money, buy
                                  input_dims=len(variables) * look_back,  # input dimensions
                                  gamma=0.9,  # discount factor of future rewards
                                  alpha=0.00025,  # learning rate for networks (actor and critic) high as its decaying at least 0.0001
                                  gae_lambda=0.9,  # lambda for generalized advantage estimation
                                  policy_clip=0.25,  # clip parameter for PPO
                                  entropy_coefficient=10,  # higher entropy coefficient encourages exploration
                                  ec_decay_rate=0.99,  # entropy coefficient decay rate
                                  batch_size=1024,  # size of the memory
                                  n_epochs=1,  # number of epochs
                                  mini_batch_size=128,  # size of the mini-batches
                                  weight_decay=0.0000005,  # weight decay
                                  l1_lambda=1e-7,  # L1 regularization lambda
                                  static_input_dims=1,  # static input dimensions (current position)
                                  lr_decay_rate=0.995,  # learning rate decay rate
                                  )

    total_rewards, episode_durations, total_balances = [], [], []
    episode_probabilities = {'train': [], 'validation': [], 'test': []}

    index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
    columns = ['Final Balance', 'Dataset Index']
    backtest_results = pd.DataFrame(index=index, columns=columns)

    window_size_2 = '1Y'
    test_rolling_datasets = rolling_window_datasets(df_test, window_size=window_size_2, look_back=look_back)
    val_rolling_datasets = rolling_window_datasets(df_validation, window_size=window_size_2, look_back=look_back)

    # Generate index labels for each rolling window dataset
    val_labels = generate_index_labels(val_rolling_datasets, 'validation')
    test_labels = generate_index_labels(test_rolling_datasets, 'test')
    all_labels = val_labels + test_labels

    # Rolling DF
    rolling_datasets = rolling_window_datasets(df_train, window_size=window_size, look_back=look_back)
    dataset_iterator = cycle(rolling_datasets)

    backtest_results, probs_dfs, balances_dfs = {}, {}, {}

    # Initialize the agent generation
    generation = 0

    for episode in tqdm(range(num_episodes)):
        start_time = time.time()

        window_df = next(dataset_iterator)
        dataset_index = episode % len(rolling_datasets)

        print(f"\nEpisode {episode + 1}: Learning from dataset with Start Date = {window_df.index.min()}, End Date = {window_df.index.max()}, len = {len(window_df)}")

        # Create a new environment with the randomly selected window's data
        env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables,
                                        tradable_markets=tradable_markets, provision=provision,
                                        initial_balance=starting_balance, leverage=leverage,
                                        reward_function=reward_calculation)

        observation = env.reset()
        done = False
        initial_balance = env.balance

        while not done:
            current_position = env.current_position
            action, prob, val = agent.choose_action(observation, env.current_position)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, prob, val, reward, done, current_position)
            observation = observation_

            # Check if enough data is collected or if the dataset ends
            if len(agent.memory.states) >= agent.memory.batch_size:
                agent.learn()
                agent.memory.clear_memory()

            if generation < agent.generation:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for df, label in zip(val_rolling_datasets + test_rolling_datasets, val_labels + test_labels):
                        future = executor.submit(BF.backtest_wrapper, 'PPO', df, agent, tradable_markets, look_back,
                                                 variables, provision, starting_balance, leverage,
                                                 Trading_Environment_Basic, reward_calculation)
                        futures.append((future, label))

                    for future, label in futures:
                        (balance, total_reward, number_of_trades, probs_df, action_df, sharpe_ratio, max_drawdown,
                         sortino_ratio, calmar_ratio, cumulative_returns, balances, provision_sum) = future.result()
                        result_data = {
                            'Agent generation': agent.generation,
                            'Label': label,
                            'Provision_sum': provision_sum,
                            'Final Balance': balance,
                            'Total Reward': total_reward,
                            'Number of Trades': number_of_trades,
                            'Sharpe Ratio': sharpe_ratio,
                            'Max Drawdown': max_drawdown,
                            'Sortino Ratio': sortino_ratio,
                            'Calmar Ratio': calmar_ratio
                        }
                        key = (agent.generation, label)

                        if key not in backtest_results:
                            backtest_results[key] = []

                        backtest_results[key].append(result_data)

                        # Store probabilities and balances for plotting
                        probs_dfs[(agent.generation, label)] = probs_df
                        balances_dfs[(agent.generation, label)] = balances

                    generation = agent.generation
                    print(f"Backtesting completed for {agent.get_name()} generation {generation}")

        # results
        end_time = time.time()
        episode_time = end_time - start_time
        total_rewards.append(env.reward_sum)
        episode_durations.append(episode_time)
        total_balances.append(env.balance)

        print(
            f"Completed learning from selected window in episode {episode + 1}: Total Reward: {env.reward_sum}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds, current Entropy Coefficient: {agent.entropy_coefficient:.2f}")
        print(f'Final Balance of Buy and Hold benchmark agent: ', starting_balance * (1 + (
                    window_df[('Close', tradable_markets)].iloc[-1] - window_df[('Close', tradable_markets)].iloc[
                look_back]) / window_df[('Close', tradable_markets)].iloc[look_back] * leverage))
        # TODO do this as cached function with benchmark agents and df as input

    # TODO repair save_model
    # prepare benchmark results
    buy_and_hold_agent = Buy_and_hold_Agent()
    sell_and_hold_agent = Sell_and_hold_Agent()

    # Run backtesting for both agents
    bah_results, _, benchmark_BAH = BF.run_backtesting(
        buy_and_hold_agent, 'BAH', val_rolling_datasets + test_rolling_datasets, val_labels + test_labels,
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, starting_balance, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    sah_results, _, benchmark_SAH = BF.run_backtesting(
        sell_and_hold_agent, 'SAH', val_rolling_datasets + test_rolling_datasets, val_labels + test_labels,
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, starting_balance, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    bah_results_prepared = prepare_backtest_results(bah_results, 'BAH')
    sah_results_prepared = prepare_backtest_results(sah_results, 'SAH')

    sah_results_prepared = sah_results_prepared.drop(('', 'Agent Generation'), axis=1)  # drop the agent generation column
    bah_results_prepared = bah_results_prepared.drop(('', 'Agent Generation'), axis=1)  # drop the agent generation column

    # Merge BAH and SAH results on 'Label'
    new_backtest_results = pd.merge(bah_results_prepared, sah_results_prepared, on=[('', 'Label')], how='outer')

    backtest_results = prepare_backtest_results(backtest_results, agent.get_name())
    backtest_results = pd.merge(backtest_results, new_backtest_results, on=[('', 'Label')], how='outer')
    backtest_results = backtest_results.set_index([('', 'Agent Generation')])

    label_series = backtest_results[('', 'Label')]
    backtest_results = backtest_results.drop(('', 'Label'), axis=1)
    backtest_results['Label'] = label_series

    print(backtest_results)

    from backtest.plots.generation_plot import plot_results, plot_total_rewards, plot_total_balances
    from backtest.plots.OHLC_probability_plot import (PnL_generation_plot, Probability_generation_plot, PnL_generations,
                                                      Reward_generations, PnL_drawdown_plot)

    plot_results(backtest_results, [(agent.get_name(), 'Final Balance'),
                                    (agent.get_name(), 'Number of Trades'), (agent.get_name(), 'Total Reward')],
                 agent.get_name())
    plot_total_rewards(total_rewards, agent.get_name())
    plot_total_balances(total_balances, agent.get_name())

    PnL_generation_plot(balances_dfs, [benchmark_BAH, benchmark_SAH], port_number=8190)
    Probability_generation_plot(probs_dfs, port_number=8191)  # TODO add here OHLC
    PnL_generations(backtest_results, port_number=8192)
    Reward_generations(backtest_results, port_number=8193)
    PnL_drawdown_plot(balances_dfs, [benchmark_BAH, benchmark_SAH], port_number=8194)

    print('end')