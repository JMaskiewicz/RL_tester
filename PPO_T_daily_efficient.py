"""
Transformer based PPO agent for trading version 4.1

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

Reward testing:
- higher penalty for wrong actions this would make agent more likely to take a neutral position
- higher number of epochs agent would less likely to take a neutral position
- premium for holding position agent would less likely to change position
"""
import cProfile
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
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ExponentialLR
import multiprocessing
from multiprocessing import Process, Queue, Event, Manager
from threading import Thread
import sys

from data.function.load_data import load_data_parallel
from data.function.rolling_window import rolling_window_datasets
from data.function.edit import process_variable
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
# import backtest.backtest_functions.functions as BF
from functions.utilis import save_actor_critic_model

"""
Reward Calculation function is the most crucial part of the RL algorithm. It is the function that determines the reward the agent receives for its actions.
"""
def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
    # Calculate the log return
    if previous_close != 0 and current_close != 0:
        log_return = math.log(current_close / previous_close)
    else:
        log_return = 0

    # Calculate the base reward
    reward = log_return * current_position * leverage

    # Penalize the agent for taking the wrong action
    if reward < 0:
        reward *= 3

    # Calculate the cost of provision if the position has changed, and it's not neutral (0).
    if current_position != previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 - provision) * 10  # penalty for changing position
    elif current_position == previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 + provision)  # small premium for holding position
    else:
        provision_cost = 0

    # Apply the provision cost
    reward += provision_cost

    # Scale the reward to enhance its significance for the learning process
    final_reward = reward * 100

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
        indices = indices[torch.randperm(n_states)]  # Shuffle indices
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        # Include static states in the generated batches
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
    def __init__(self, n_actions, input_dims, n_heads=4, n_layers=3, dropout_rate=1 / 8, static_input_dims=1):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.static_input_dims = static_input_dims
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        encoder_layers = TransformerEncoderLayer(d_model=input_dims, nhead=n_heads, dropout=dropout_rate,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)

        self.max_position_embeddings = 512
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.max_position_embeddings, input_dims))
        self.fc_static = nn.Linear(static_input_dims, input_dims)

        self.fc1 = nn.Linear(input_dims * 2, 2048)
        self.ln1 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)
        self.fc5 = nn.Linear(256, 128)
        self.ln5 = nn.LayerNorm(128)
        self.fc6 = nn.Linear(128, n_actions)

        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dynamic_state, static_state):
        batch_size, seq_length, _ = dynamic_state.size()
        positional_encoding = self.positional_encoding[:, :seq_length, :].expand(batch_size, -1, -1)

        dynamic_state = dynamic_state + positional_encoding
        transformer_out = self.transformer_encoder(dynamic_state)

        static_state_encoded = self.fc_static(static_state.unsqueeze(1))
        combined_features = torch.cat((transformer_out[:, -1, :], static_state_encoded.squeeze(1)), dim=1)

        x = self.relu(self.ln1(self.fc1(combined_features)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.relu(self.ln4(self.fc4(x)))
        x = self.relu(self.ln5(self.fc5(x)))
        x = self.fc6(x)

        return self.softmax(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_heads=4, n_layers=3, dropout_rate=1 / 8, static_input_dims=1):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.static_input_dims = static_input_dims
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        encoder_layers = TransformerEncoderLayer(d_model=input_dims, nhead=n_heads, dropout=dropout_rate,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)

        self.max_position_embeddings = 512
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.max_position_embeddings, input_dims))
        self.fc_static = nn.Linear(static_input_dims, input_dims)

        self.fc1 = nn.Linear(input_dims * 2, 2048)
        self.ln1 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)
        self.fc5 = nn.Linear(256, 128)
        self.ln5 = nn.LayerNorm(128)
        self.fc6 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, dynamic_state, static_state):
        batch_size, seq_length, _ = dynamic_state.size()
        positional_encoding = self.positional_encoding[:, :seq_length, :].expand(batch_size, -1, -1)

        dynamic_state = dynamic_state + positional_encoding
        transformer_out = self.transformer_encoder(dynamic_state)

        static_state_encoded = self.fc_static(static_state.unsqueeze(1))
        combined_features = torch.cat((transformer_out[:, -1, :], static_state_encoded.squeeze(1)), dim=1)

        x = self.relu(self.ln1(self.fc1(combined_features)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.relu(self.ln4(self.fc4(x)))
        x = self.relu(self.ln5(self.fc5(x)))
        x = self.fc6(x)

        return x


class Transformer_PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024,
                 n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5,
                 static_input_dims=1, lr_decay_rate=0.99):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Not sure why CPU is faster
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.gamma = gamma  # Discount factor
        self.policy_clip = policy_clip  # PPO policy clipping parameter
        self.n_epochs = n_epochs  # Number of optimization epochs per batch
        self.gae_lambda = gae_lambda  # Generalized Advantage Estimation lambda
        self.mini_batch_size = mini_batch_size  # Size of mini-batches for optimization
        self.entropy_coefficient = entropy_coefficient  # Entropy coefficient for encouraging exploration
        self.l1_lambda = l1_lambda  # L1 regularization coefficient
        self.lr_decay_rate = lr_decay_rate  # Learning rate decay rate

        # Initialize the actor and critic networks with static input dimensions
        self.actor = ActorNetwork(n_actions, input_dims, static_input_dims=static_input_dims).to(self.device)
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
        print("-" * 50)
        # Set the actor and critic networks to training mode
        self.actor.train()
        self.critic.train()

        # Stack the tensors in the memory
        self.memory.stack_tensors()

        # Loop through the optimization epochs
        for _ in range(self.n_epochs):
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
            advantages, discounted_rewards = self.compute_discounted_rewards(reward_arr, vals_arr.cpu().numpy(), dones_arr)
            advantages = advantages.clone().detach().to(self.device)  #
            discounted_rewards = discounted_rewards.clone().detach().to(self.device)

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
                new_probs, dist_entropy, actor_loss, critic_loss = self.calculate_loss(batch_states, batch_actions,
                                                                                       batch_old_probs,
                                                                                       batch_advantages, batch_returns,
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

        # track the time it takes to learn
        end_time = time.time()
        episode_time = end_time - start_time

        # print the time it takes to learn
        print(f"Learning of agent generation {self.generation} completed in {episode_time} seconds")
        print("-" * 50)


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
        # Calculate advantages and discounted returns
        n = len(rewards)

        # Create tensors to store advantages and discounted returns
        discounted_rewards = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Initialize the advantage and the last GAE (Generalized Advantage Estimation) lambda
        last_gae_lam = 0

        # Convert 'dones' to a float tensor
        dones = dones.float()

        for t in reversed(range(n)):
            # If the current time step is the last one, the next non-terminal is 0
            if t == n - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0
            # Otherwise, the next non-terminal is 1 - 'dones[t + 1]', and the next value is 'values[t + 1]'
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            # Calculate the Temporal Difference (TD) error
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]

            # Update the advantages and the last GAE lambda
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            # Update the advantages and the discounted returns
            advantages[t] = last_gae_lam

            # Calculate the discounted return
            discounted_rewards[t] = advantages[t] + values[t]

        return advantages, discounted_rewards

    @torch.no_grad()
    def choose_action(self, observation, static_input):  # TODO check if this is correct
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


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001,
                 initial_balance=10000, leverage=1):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)  # Reset the index of the DataFrame
        self.look_back = look_back  # Number of time steps to look back
        self.initial_balance = initial_balance  # Initial balance
        self.reward_sum = 0  # Initialize the reward sum
        self.current_position = 0  # This is a static part of the state
        self.variables = variables  # List of variables to be used in the environment
        self.tradable_markets = tradable_markets  # asset to be traded in the environment
        self.provision = provision  # Provision cost
        self.leverage = leverage  # Leverage

        # Define action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.Discrete(3)

        # Reset the environment to initialize the state
        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        return input_dims

    def reset(self, observation_idx=None, reset_position=True):
        # Reset the environment to the initial state
        if observation_idx is not None:
            self.current_step = observation_idx + self.look_back
        else:
            self.current_step = self.look_back

        # Reset the balance and reward sum
        self.balance = self.initial_balance
        self.reward_sum = 0

        # Reset the current position
        if reset_position:
            self.current_position = 0

        # Set the done flag to False
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        # Get the observation array for the current time step
        start = max(self.current_step - self.look_back, 0)
        end = self.current_step

        # Create a list of the observation arrays for each variable
        tasks = [(self.df[variable['variable']].iloc[start:end].values, variable['edit']) for variable in
                 self.variables]

        # Use ThreadPoolExecutor to parallelize data transformation
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Prepare and execute tasks
            future_to_variable = {executor.submit(process_variable, data, edit_type): (data, edit_type) for data, edit_type in tasks}
            results = []
            for future in concurrent.futures.as_completed(future_to_variable):
                data, edit_type = future_to_variable[future]  # Get the original data and edit type
                try:
                    result = future.result()  # Get the result of the transformation
                except Exception as exc:
                    print('%r generated an exception: %s' % ((data, edit_type), exc))  # Print exception
                else:
                    results.append(result)  # Append the result to the list of results

        # Concatenate results to form the scaled observation array
        scaled_observation = np.concatenate(results).flatten()
        return scaled_observation

    def step(self, action):  # TODO: Check if this is correct
        action_mapping = {0: -1, 1: 0, 2: 1}  # best mapping ever :)
        mapped_action = action_mapping[action]  # Map the action to a position change

        # Get the current price and the price of the next time step for the reward calculation and PnL
        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step + 1]

        # balance update
        market_return = next_price / current_price - 1

        # Provision cost calculation if the position has changed
        if mapped_action != self.current_position:
            provision_cost = self.provision * (abs(mapped_action) == 1)
        else:
            provision_cost = 0

        # Update the balance
        self.balance *= (1 + market_return * self.current_position * self.leverage - provision_cost)

        # reward calculation with reward function on the top of the file (reward_calculation)
        final_reward = reward_calculation(current_price, next_price, self.current_position, mapped_action, self.leverage, self.provision)
        self.reward_sum += final_reward  # Update the reward sum
        self.current_position = mapped_action  # Update the current position
        self.current_step += 1  # Increment the current step

        # Check if the episode is done
        self.done = self.current_step >= len(self.df) - 1

        return self._next_observation(), final_reward, self.done, {}

"""
Description of parallelization of the environment
Initiate the learning phase as soon as enough experiences are collected, then immediately resume experience gathering after learning while backtesting is performed in parallel for the updated agent. 
This approach ensures continuous data collection and efficient utilization of computational resources.
"""

def generate_predictions_and_backtest_AC(df, agent, mkf, look_back, variables, provision=0.001, starting_balance=10000, leverage=1, Trading_Environment_Basic=None):
    """
    # TODO add description
    """
    action_probabilities_list = []
    best_action_list = []
    balances = []
    number_of_trades = 0

    # Preparing the environment
    agent.actor.eval()
    agent.critic.eval()

    with torch.no_grad():
        # Create a backtesting environment
        env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                        tradable_markets=mkf, provision=provision,
                                        initial_balance=starting_balance, leverage=leverage)

        observation = env.reset()
        done = False

        while not done:  # TODO check if this is correct
            action_probs = agent.get_action_probabilities(observation, env.current_position)
            best_action = np.argmax(action_probs)

            if (best_action-1) != env.current_position and abs(best_action-1) == 1:
                number_of_trades += 1

            observation_, reward, done, info = env.step(best_action)
            observation = observation_

            balances.append(env.balance)  # Update balances
            action_probabilities_list.append(action_probs.tolist())
            best_action_list.append(best_action-1)

    # KPI Calculations
    buy_and_hold_return = starting_balance * (df[('Close', mkf)].iloc[-1] / df[('Close', mkf)].iloc[look_back] - 1)
    sell_and_hold_return = starting_balance * (1 - df[('Close', mkf)].iloc[-1] / df[('Close', mkf)].iloc[look_back])

    returns = pd.Series(balances).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(df)-env.look_back) if returns.std() > 1e-6 else float('nan')  # change 252

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    negative_volatility = returns[returns < 0].std() * np.sqrt(len(df)-env.look_back)  # change 252
    sortino_ratio = returns.mean() / negative_volatility if negative_volatility > 1e-6 else float('nan')

    annual_return = cumulative_returns.iloc[-1] ** ((len(df)-env.look_back) / len(returns)) - 1  # change 252
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else float('nan')

    # Convert the list of action probabilities to a DataFrame
    probabilities_df = pd.DataFrame(action_probabilities_list, columns=['Short', 'Neutral', 'Long'])
    action_df = pd.DataFrame(best_action_list, columns=['Action'])

    # Ensure the agent's networks are reverted back to training mode
    agent.actor.train()
    agent.critic.train()

    return env.balance, env.reward_sum, number_of_trades, probabilities_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances

def backtest_wrapper_AC(df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic=None):
    """
    # TODO add description
    AC - Actor Critic
    """
    return generate_predictions_and_backtest_AC(df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic)

def backtest_in_background(agent, backtest_results, num_workers_backtesting, val_rolling_datasets, test_rolling_datasets, val_labels, test_labels, probs_dfs, balances_dfs, backtesting_completed):
    start_time = time.time()
    print("Starting backtesting...")
    with ThreadPoolExecutor(max_workers=num_workers_backtesting) as executor:
        futures = []
        for df, label in zip(val_rolling_datasets + test_rolling_datasets, val_labels + test_labels):
            future = executor.submit(backtest_wrapper_AC, df, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage, Trading_Environment_Basic)
            futures.append((future, label))

        for future, label in futures:
            (balance, total_reward, number_of_trades, probs_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances) = future.result()
            result_data = {
                'Agent generation': agent.generation,
                'Label': label,
                'Final Balance': balance,
                'Total Reward': total_reward,
                'Number of Trades': number_of_trades,
                'Buy and Hold Return': buy_and_hold_return,
                'Sell and Hold Return': sell_and_hold_return,
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

    # Signal that backtesting is completed
    backtesting_completed.set()
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"Backtesting completed in {episode_time:.2f} seconds\n")

def environment_worker(dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signal, resume_signal, total_rewards, total_balances, worker_id, individual_worker_batch_size, workers_completed, workers_completed_signal, shared_episodes_counter):
    random.seed(worker_id)  # Seed the random number generator with a unique seed for this worker

    experiences_collected = 0  # Initialize a counter for collected experiences
    start_time = time.time()  # Record the start time of data collection

    for episode in range(max_episodes_per_worker):
        shared_episodes_counter.value += 1
        df = random.choice(dfs)
        env = Trading_Environment_Basic(df, **env_settings)
        observation = env.reset()
        done = False

        while not done:
            work_event.wait()  # Wait for permission to work
            if resume_signal.is_set():
                print(f"Worker {multiprocessing.current_process().name}: Starting...")
                experiences_collected = 0  # Reset the counter after pausing
                start_time = time.time()  # Reset the start time for the next batch
                resume_signal.clear()  # Clear the resume signal

            action, prob, val = agent.choose_action(observation, env.current_position)
            observation_, reward, done, info = env.step(action)
            experience = (observation, action, prob, val, reward, done, env.current_position)
            shared_queue.put(experience)
            observation = observation_

            experiences_collected += 1  # Increment the counter for each experience collected

            if experiences_collected >= individual_worker_batch_size:
                end_time = time.time()  # Record the time when the batch size limit is reached
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                print(f"Worker {multiprocessing.current_process().name}: Reached individual batch size limit in {elapsed_time:.2f} seconds.")
                pause_signal.set()  # Signal this worker to pause
                # Wait for the main process or a manager function to clear this worker's pause signal.
                while pause_signal.is_set():
                    time.sleep(0.1)  # Sleep to prevent busy waiting

        # Append the total reward and final balance for this episode to the shared lists
        total_rewards.append(env.reward_sum)
        total_balances.append(env.balance)
        print(f"Worker {multiprocessing.current_process().name} completed training df of length {len(df)}, first observation in training df is {df.index[0]}, episode {episode+1}/{max_episodes_per_worker} with cumulative reward {env.reward_sum} and final balance {env.balance}")

    print(f"Worker {multiprocessing.current_process().name} has completed all tasks.")
    workers_completed.value += 1
    if workers_completed.value >= 1:
        workers_completed_signal.set()

# TODO add description
# TODO add early stopping based on the validation set from the backtesting
def collect_and_learn(dfs, max_episodes_per_worker, env_settings, batch_size_for_learning, backtest_results, agent,
                      num_workers, num_workers_backtesting, backtesting_frequency=1):

    manager = Manager()
    total_rewards, total_balances, shared_episodes_counter, workers_completed, backtesting_completed, work_event, pause_signals, resume_signals, workers_completed_signal, shared_queue = setup_shared_resources_and_events(
        manager, num_workers)

    workers = start_workers(num_workers, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event,
                            pause_signals, resume_signals, total_rewards, total_balances, workers_completed,
                            workers_completed_signal, shared_episodes_counter, batch_size_for_learning)

    manage_learning_and_backtesting(agent, num_workers_backtesting, backtest_results, backtesting_completed, work_event,
                                    pause_signals, resume_signals, shared_queue, workers_completed_signal,
                                    shared_episodes_counter, total_rewards, total_balances, batch_size_for_learning,
                                    backtesting_frequency)

    for worker in workers:
        worker.join()

    print("\r" + " " * 100, end='')
    print("All workers stopped.")
    return list(total_rewards), list(total_balances)

def manage_learning_and_backtesting(agent, num_workers_backtesting, backtest_results, backtesting_completed, work_event, pause_signals, resume_signals, shared_queue, workers_completed_signal, shared_episodes_counter, total_rewards, total_balances, batch_size_for_learning, backtesting_frequency):
    agent_generation = 0
    total_experiences = 0
    start_time = time.time()
    try:
        while True:
            if not work_event.is_set():
                work_event.set()

            while not shared_queue.empty():
                experience = shared_queue.get_nowait()
                agent.store_transition(*experience)
                total_experiences += 1

            if total_experiences >= batch_size_for_learning and backtesting_completed.is_set():
                print("\nLearning phase initiated.")
                agent.learn()  # Placeholder for your agent's learning method
                total_experiences = 0
                # agent.memory.clear_memory() # Uncomment if your agent has a method to clear memory

                for signal in pause_signals:
                    signal.clear()
                work_event.set()
                for resume_signal in resume_signals:
                    resume_signal.set()

                if agent.generation > agent_generation and agent.generation % backtesting_frequency == 0:
                    backtesting_completed.clear()
                    backtest_thread = Thread(target=backtest_in_background, args=(agent, backtest_results, num_workers_backtesting, val_rolling_datasets, test_rolling_datasets, val_labels, test_labels, probs_dfs, balances_dfs, backtesting_completed))
                    backtest_thread.start()
                    agent_generation = agent.generation

            # Progress update is within the while loop to continuously update
            current_episodes = shared_episodes_counter.value
            total_episodes = max_episodes_per_worker * num_workers
            progress_percentage = (current_episodes / total_episodes) * 100
            bar_length = 20
            filled_length = int(round(bar_length * progress_percentage / 100))
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            elapsed_time = time.time() - start_time
            speed = current_episodes / elapsed_time if elapsed_time > 0 else float('inf')
            eta_seconds = ((total_episodes - current_episodes) / speed) if speed > 0 else float('inf')
            formatted_elapsed_time = format_time(elapsed_time)
            formatted_eta = format_time(eta_seconds)
            sys.stdout.write(
                f"\033[92m\rProgress: {progress_percentage:.0f}% |{bar}| {current_episodes}/{total_episodes} [Elapsed: {formatted_elapsed_time}, ETA: {formatted_eta}, Speed: {speed:.2f}it/s]\033[0m")
            sys.stdout.flush()

            if workers_completed_signal.is_set():
                backtesting_completed.wait()
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Final progress update after loop completion or interruption
    print("\r" + " " * 100, end='')  # Clear the line
    print(f"\033[92m\rProgress: 100% |{'█' * bar_length}| {max_episodes_per_worker}/{max_episodes_per_worker * num_workers} [{elapsed_time:.2f}s, {speed:.2f}it/s]\033[0m")
    print("All workers stopped.")

def setup_shared_resources_and_events(manager, num_workers):
    total_rewards = manager.list()
    total_balances = manager.list()
    shared_episodes_counter = manager.Value('i', 0)
    workers_completed = manager.Value('i', 0)
    backtesting_completed = Event()
    backtesting_completed.set()
    work_event = Event()
    pause_signals = [Event() for _ in range(num_workers)]
    resume_signals = [Event() for _ in range(num_workers)]
    workers_completed_signal = Event()
    shared_queue = manager.Queue()
    return total_rewards, total_balances, shared_episodes_counter, workers_completed, backtesting_completed, work_event, pause_signals, resume_signals, workers_completed_signal, shared_queue

def start_workers(num_workers, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signals, resume_signals, total_rewards, total_balances, workers_completed, workers_completed_signal, shared_episodes_counter, batch_size_for_learning):
    workers = []
    for i in range(num_workers):
        worker_process = Process(target=environment_worker, args=(dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signals[i], resume_signals[i], total_rewards, total_balances, i+1, batch_size_for_learning // num_workers, workers_completed, workers_completed_signal, shared_episodes_counter))
        worker_process.start()
        workers.append(worker_process)
    return workers

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h:{m}m:{s}s"


def generate_index_labels(rolling_datasets, dataset_type):
    index_labels = []
    for dataset in rolling_datasets:
        last_day = dataset.index[-1].strftime('%Y-%m-%d')
        label = f"{dataset_type}_{last_day}"
        index_labels.append(label)
    return index_labels

def prepare_backtest_results(backtest_results):
    agent_generations = []
    labels = []
    final_balances = []
    total_rewards = []
    number_of_trades = []
    buy_and_hold_returns = []
    sell_and_hold_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    sortino_ratios = []
    calmar_ratios = []

    # Iterating over each record in the dataset
    for (agent_gen, label), metrics in backtest_results.items():
        for metric in metrics:
            agent_generations.append(agent_gen)
            labels.append(label)
            final_balances.append(metric['Final Balance'])
            total_rewards.append(metric['Total Reward'])
            number_of_trades.append(metric['Number of Trades'])
            buy_and_hold_returns.append(metric['Buy and Hold Return'])
            sell_and_hold_returns.append(metric['Sell and Hold Return'])
            sharpe_ratios.append(metric['Sharpe Ratio'])
            max_drawdowns.append(metric['Max Drawdown'])
            sortino_ratios.append(metric['Sortino Ratio'])
            calmar_ratios.append(metric['Calmar Ratio'])

    # Creating a DataFrame from the lists
    df = pd.DataFrame({
        'Agent Generation': agent_generations,
        'Label': labels,
        'Final Balance': final_balances,
        'Total Reward': total_rewards,
        'Number of Trades': number_of_trades,
        'Buy and Hold Return': buy_and_hold_returns,
        'Sell and Hold Return': sell_and_hold_returns,
        'Sharpe Ratio': sharpe_ratios,
        'Max Drawdown': max_drawdowns,
        'Sortino Ratio': sortino_ratios,
        'Calmar Ratio': calmar_ratios,
    })
    return df

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
        {"indicator": "ATR", "mkf": "EURUSD", "length": 24},
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

    df['Returns_Close', 'EURUSD'] = (df['Returns_Close', 'EURUSD'])
    df['Returns_Close', 'USDJPY'] = (df['Returns_Close', 'USDJPY'])
    df['Returns_Close', 'EURJPY'] = (df['Returns_Close', 'EURJPY'])
    df['Returns_Close', 'GBPUSD'] = (df['Returns_Close', 'GBPUSD'])

    add_time_sine_cosine(df, '1D')
    df[("sin_time_1D", "")] = df[("sin_time_1D", "")] / 2 + 0.5
    df[("cos_time_1D", "")] = df[("cos_time_1D", "")] / 2 + 0.5
    df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")] / 100

    df = df.dropna()
    # data before 2006 has some missing values ie gaps in the data, also in march, april 2023 there are some gaps
    start_date = '2008-01-01'  # worth to keep 2008 as it was a financial crisis
    validation_date = '2021-01-01'
    test_date = '2022-01-01'
    df_train = df[start_date:validation_date]
    df_validation = df[validation_date:test_date]
    df_test = df[test_date:'2023-01-01']

    variables = [
        {"variable": ("Close", "USDJPY"), "edit": "normalize"},
        {"variable": ("Close", "EURUSD"), "edit": "normalize"},
        {"variable": ("Close", "EURJPY"), "edit": "normalize"},
        {"variable": ("Close", "GBPUSD"), "edit": "normalize"},
        {"variable": ("RSI_14", "EURUSD"), "edit": None},
        {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
        #{"variable": ("sin_time_1D", ""), "edit": None},
        #{"variable": ("cos_time_1D", ""), "edit": None},
        #{"variable": ("Returns_Close", "EURUSD"), "edit": None},
        #{"variable": ("Returns_Close", "USDJPY"), "edit": None},
        #{"variable": ("Returns_Close", "EURJPY"), "edit": None},
        #{"variable": ("Returns_Close", "GBPUSD"), "edit": None},
    ]

    tradable_markets = 'EURUSD'
    window_size = '3M'
    starting_balance = 10000
    look_back = 20
    # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001
    provision = 0.0001  # 0.001, cant be too high as it would not learn to trade

    # Training parameters
    batch_size = 1024  # 8192
    epochs = 20  # 40
    mini_batch_size = 128  # 256
    leverage = 10  # 30
    l1_lambda = 1e-7  # L1 regularization
    weight_decay = 0.00001  # L2 regularization

    # Number of CPU cores for number of workers
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    num_workers = min(max(1, num_cores - 1), 8)  # Number of workers, some needs to left for backtesting
    num_workers_backtesting = 12  # backtesting is parallelized in same time that gathering data for next generation
    num_episodes = 40  # need to be divisible by num_workers
    max_episodes_per_worker = num_episodes // num_workers

    '''
    Number of workers need to be related to window sizes to have best performance
    Backtesting is bottleneck now maybe its better to backtest only each 10th generation
    '''

    # Split validation and test datasets into multiple rolling windows
    window_size_2 = '3M'
    test_rolling_datasets = rolling_window_datasets(df_test, window_size=window_size_2, look_back=look_back)
    val_rolling_datasets = rolling_window_datasets(df_validation, window_size=window_size_2, look_back=look_back)

    # Generate index labels for each rolling window dataset
    val_labels = generate_index_labels(val_rolling_datasets, 'validation')
    test_labels = generate_index_labels(test_rolling_datasets, 'test')
    all_labels = val_labels + test_labels

    # Create a DataFrame to hold backtesting results for all rolling windows
    columns = ['Agent generation', 'Label', 'Final Balance', 'Number of Trades']
    backtest_results = {}

    # Create an instance of the agent
    agent = Transformer_PPO_Agent(n_actions=3,  # sell, hold money, buy
                                  input_dims=len(variables) * look_back,  # input dimensions
                                  gamma=0.975,  # discount factor for future rewards
                                  alpha=0.0001,  # learning rate for networks (actor and critic) high as its decaying
                                  gae_lambda=0.9,  # lambda for generalized advantage estimation
                                  policy_clip=0.2,  # clip parameter for PPO
                                  entropy_coefficient=0.5,  # higher entropy coefficient encourages exploration
                                  batch_size=batch_size,  # size of the memory
                                  n_epochs=epochs,  # number of epochs
                                  mini_batch_size=mini_batch_size,  # size of the mini-batches
                                  weight_decay=weight_decay,  # weight decay
                                  l1_lambda=l1_lambda,  # L1 regularization lambda
                                  static_input_dims=1,  # static input dimensions (current position)
                                  lr_decay_rate=0.99,  # learning rate decay rate
                                  )

    # Environment settings
    env_settings = {
        'look_back': look_back,
        'variables': variables,
        'tradable_markets': tradable_markets,
        'provision': provision,
        'initial_balance': starting_balance,
        'leverage': leverage
    }

    # Rolling DF
    rolling_datasets = rolling_window_datasets(df_train, window_size=window_size, look_back=look_back)
    dataset_iterator = cycle(rolling_datasets)

    probs_dfs = {}
    balances_dfs = {}

    # Collecting and learning data in parallel
    total_rewards, total_balances = collect_and_learn(rolling_datasets, max_episodes_per_worker, env_settings, batch_size, backtest_results, agent, num_workers, num_workers_backtesting, backtesting_frequency=5)
    backtest_results = prepare_backtest_results(backtest_results)
    backtest_results = backtest_results.set_index(['Agent Generation'])
    print(df)

    from backtest.plots.generation_plot import plot_results, plot_total_rewards, plot_total_balances
    from backtest.plots.OHLC_probability_plot import PnL_generation_plot, Probability_generation_plot

    plot_results(backtest_results, ['Final Balance', 'Number of Trades', 'Total Reward'], agent.get_name())
    plot_total_rewards(total_rewards, agent.get_name())
    plot_total_balances(total_balances, agent.get_name())

    PnL_generation_plot(balances_dfs, port_number=8050)
    Probability_generation_plot(probs_dfs, port_number=8051)

    end_time_X = time.time()
    episode_time_X = end_time_X - start_time_X
    print(f"full run completed in {episode_time_X:.2f} seconds, END")

