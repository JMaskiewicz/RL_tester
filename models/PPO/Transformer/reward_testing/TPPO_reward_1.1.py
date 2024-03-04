"""
Transformer based PPO agent for trading version 1.6

# TODO LIST
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- HRL (Hierarchical Reinforcement Learning): Use hierarchical reinforcement learning to learn sub-policies for different tasks. Master agent would distribute capital among sub-agents for different tickers.


Some notes on the code:
- learning of the agent is fast (3.38s for batch of 8192 and mini-batch of 256)
- but the backtesting and the generation of actions is slow ie "while not done:" in the learning function

Reward testing:
- higher penalty for wrong actions
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
import concurrent.futures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ExponentialLR

from data.function.load_data import load_data_parallel
from data.function.rolling_window import rolling_window_datasets
#from data.function.edit import normalize_data, standardize_data
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
# import backtest.backtest_functions.functions as BF
from functions.utilis import save_actor_critic_model

def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
    reward = 0
    reward_return = (current_close / previous_close - 1) * current_position * leverage
    if previous_position != current_position:
        provision_cost = provision * (abs(current_position) == 1)
    else:
        provision_cost = 0

    if reward_return < 0:
        reward_return = 3 * reward_return

    reward += reward_return * 1000 - provision_cost * 100

    return reward

def generate_predictions_and_backtest_AC(df, agent, mkf, look_back, variables, provision=0.001, initial_balance=10000, leverage=1, Trading_Environment_Basic=None, plot=False):
    # Initial setup
    balance = initial_balance
    total_reward = 0
    number_of_trades = 0
    balances = [initial_balance]

    action_probabilities_list = []
    best_action_list = []

    # Preparing the environment
    env = Trading_Environment_Basic(df, look_back=look_back, variables=variables, tradable_markets=[mkf],
                                     provision=provision, initial_balance=initial_balance, leverage=leverage)
    agent.actor.eval()
    agent.critic.eval()

    with torch.no_grad():  # Disable gradient computation
        for observation_idx in range(len(df) - env.look_back):
            # Environment observation
            current_position = env.current_position
            observation = env.reset(observation_idx, reset_position=False)

            # Get action probabilities for the current observation
            action_probs = agent.get_action_probabilities(observation, current_position)
            best_action = np.argmax(action_probs) - 1  # Adjusting action scale if necessary

            # Simulate the action in the environment
            if observation_idx + env.look_back < len(df):
                current_price = df[('Close', mkf)].iloc[observation_idx + env.look_back - 1]
                next_price = df[('Close', mkf)].iloc[observation_idx + env.look_back]
                market_return = next_price / current_price - 1

                if best_action != current_position:
                    provision_cost = provision * (abs(best_action) == 1)
                    number_of_trades += 1
                else:
                    provision_cost = 0
                # TODO there is something wrong balance is only decreasing ???
                balance *= (1 + market_return * current_position * leverage - provision_cost)
                balances.append(balance)  # Update daily balance

                # Store action probabilities
                action_probabilities_list.append(action_probs.tolist())
                best_action_list.append(best_action)
                current_position = best_action

                total_reward += reward_calculation(current_price, next_price, current_position, best_action, leverage, provision)

    # KPI Calculations
    buy_and_hold_return = np.log(df[('Close', mkf)].iloc[-1] / df[('Close', mkf)].iloc[env.look_back])
    sell_and_hold_return = -buy_and_hold_return

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

    return balance, total_reward, number_of_trades, probabilities_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances

def backtest_wrapper_AC(df, agent, mkf, look_back, variables, provision, initial_balance, leverage,
                        Trading_Environment_Basic=None):
    """
    # TODO add description
    AC - Actor Critic
    """
    return generate_predictions_and_backtest_AC(df, agent, mkf, look_back, variables, provision, initial_balance,
                                                leverage, Trading_Environment_Basic)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.entropy_coefficient = entropy_coefficient
        self.l1_lambda = l1_lambda
        self.lr_decay_rate = lr_decay_rate

        # Initialize the actor and critic networks with static input dimensions
        self.actor = ActorNetwork(n_actions, input_dims, static_input_dims=static_input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims, static_input_dims=static_input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=self.lr_decay_rate)
        self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=self.lr_decay_rate)

        self.memory = PPOMemory(batch_size, self.device)

        self.generation = 0

    def store_transition(self, state, action, probs, vals, reward, done, static_state):
        # Include static_state in the memory storage
        self.memory.store_memory(state, action, probs, vals, reward, done, static_state)

    def learn(self):
        start_time = time.time()
        self.actor.train()
        self.critic.train()

        self.memory.stack_tensors()

        for _ in range(self.n_epochs):
            # Generating the data for the entire batch, including static states
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, static_states_arr, batches = self.memory.generate_batches()

            # Convert arrays to tensors and move to the device
            state_arr = state_arr.clone().detach().to(self.device)
            action_arr = action_arr.clone().detach().to(self.device)
            old_prob_arr = old_prob_arr.clone().detach().to(self.device)
            vals_arr = vals_arr.clone().detach().to(self.device)
            reward_arr = reward_arr.clone().detach().to(self.device)
            dones_arr = dones_arr.clone().detach().to(self.device)
            static_states_arr = static_states_arr.clone().detach().to(self.device)  # Static states

            # Compute advantages and discounted rewards
            advantages, discounted_rewards = self.compute_discounted_rewards(reward_arr, vals_arr.cpu().numpy(),
                                                                             dones_arr)
            advantages = advantages.clone().detach().to(self.device)
            discounted_rewards = discounted_rewards.clone().detach().to(self.device)

            # Creating mini-batches and training
            num_samples = len(state_arr)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.mini_batch_size):
                minibatch_indices = indices[start_idx:start_idx + self.mini_batch_size]

                batch_states = state_arr[minibatch_indices].clone().detach().to(self.device)
                batch_actions = action_arr[minibatch_indices].clone().detach().to(self.device)
                batch_old_probs = old_prob_arr[minibatch_indices].clone().detach().to(self.device)
                batch_advantages = advantages[minibatch_indices].clone().detach().to(self.device)
                batch_returns = discounted_rewards[minibatch_indices].clone().detach().to(self.device)
                batch_static_states = static_states_arr[minibatch_indices].clone().detach().to(
                    self.device)  # Static states for the batch

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Calculate actor and critic losses, include static states in forward passes
                new_probs, dist_entropy, actor_loss, critic_loss = self.calculate_loss(batch_states, batch_actions,
                                                                                       batch_old_probs,
                                                                                       batch_advantages, batch_returns,
                                                                                       batch_static_states)

                # Perform backpropagation and optimization steps
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
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Learning of agent generation {self.generation} completed in {episode_time} seconds")

    def calculate_loss(self, batch_states, batch_actions, batch_old_probs, batch_advantages, batch_returns,
                       batch_static_states):
        if batch_states.dim() == 2:
            batch_states = batch_states.unsqueeze(1)

        # Actor loss calculations
        new_probs = self.actor(batch_states, batch_static_states)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(batch_actions)
        prob_ratios = torch.exp(new_log_probs - batch_old_probs)
        surr1 = prob_ratios * batch_advantages
        surr2 = torch.clamp(prob_ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * dist.entropy().mean()

        # Critic loss calculations
        critic_values = self.critic(batch_states, batch_static_states).squeeze(-1)
        critic_loss = (batch_returns - critic_values).pow(2).mean()

        return new_probs, dist.entropy(), actor_loss, critic_loss

    def compute_discounted_rewards(self, rewards, values, dones):
        n = len(rewards)
        discounted_rewards = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        # Convert 'dones' to a float tensor
        dones = dones.float()

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            discounted_rewards[t] = advantages[t] + values[t]

        return advantages, discounted_rewards

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
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0  # This is a static part of the state
        self.variables = variables
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage

        # Define action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        return input_dims

    def reset(self, observation_idx=None, reset_position=True):
        if observation_idx is not None:
            self.current_step = observation_idx + self.look_back
        else:
            self.current_step = self.look_back

        self.balance = self.initial_balance
        if reset_position:
            self.current_position = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        start = max(self.current_step - self.look_back, 0)
        end = self.current_step

        tasks = [(self.df[variable['variable']].iloc[start:end].values, variable['edit']) for variable in
                 self.variables]

        # Use ThreadPoolExecutor to parallelize data transformation
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Prepare and execute tasks
            future_to_variable = {executor.submit(process_variable, data, edit_type): (data, edit_type) for data, edit_type in tasks}
            results = []
            for future in concurrent.futures.as_completed(future_to_variable):
                data, edit_type = future_to_variable[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % ((data, edit_type), exc))
                else:
                    results.append(result)

        # Concatenate results to form the scaled observation array
        scaled_observation = np.concatenate(results).flatten()
        return scaled_observation

    def step(self, action):  # TODO: Check if this is correct
        action_mapping = {0: -1, 1: 0, 2: 1}
        mapped_action = action_mapping[action]

        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step + 1]

        # balance update
        market_return = next_price / current_price - 1
        if mapped_action != self.current_position:
            provision_cost = self.provision * (abs(mapped_action) == 1)
        else:
            provision_cost = 0

        self.balance *= (1 + market_return * self.current_position * self.leverage - provision_cost)

        # reward calculation
        final_reward = reward_calculation(current_price, next_price, self.current_position, mapped_action, self.leverage, self.provision)
        self.current_position = mapped_action
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1

        return self._next_observation(), final_reward, self.done, {}

from numba import jit

# this speed up calculations by 10% (3s per episode)
@jit(nopython=True)
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

@jit(nopython=True)
def standardize_data(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    standardized = (data - mean_val) / std_val
    return standardized

def process_variable(data, edit_type):
    if edit_type == 'standardize':
        return standardize_data(data)
    elif edit_type == 'normalize':
        return normalize_data(data)
    else:
        return data

if __name__ == '__main__':
    # Example usage
    # Stock market variables
    df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD'], '1H')

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

    df['Returns_Close', 'EURUSD'] = (df['Returns_Close', 'EURUSD'] * 10 + 0.5)
    df['Returns_Close', 'USDJPY'] = (df['Returns_Close', 'USDJPY'] * 10 + 0.5)
    df['Returns_Close', 'EURJPY'] = (df['Returns_Close', 'EURJPY'] * 10 + 0.5)
    df['Returns_Close', 'GBPUSD'] = (df['Returns_Close', 'GBPUSD'] * 10 + 0.5)

    add_time_sine_cosine(df, '1D')
    df[("sin_time_1D", "")] = df[("sin_time_1D", "")] / 2 + 0.5
    df[("cos_time_1D", "")] = df[("cos_time_1D", "")] / 2 + 0.5
    df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")] / 100

    df = df.dropna()
    # data before 2006 has some missing values ie gaps in the data
    start_date = '2008-01-01'
    validation_date = '2022-01-01'
    test_date = '2023-01-01'
    df_train = df[start_date:validation_date]
    df_validation = df[validation_date:test_date]
    df_test = df[test_date:]
    variables = [
        {"variable": ("Close", "USDJPY"), "edit": "normalize"},
        {"variable": ("Close", "EURUSD"), "edit": "normalize"},
        {"variable": ("Close", "EURJPY"), "edit": "normalize"},
        {"variable": ("Close", "GBPUSD"), "edit": "normalize"},
        {"variable": ("RSI_14", "EURUSD"), "edit": None},
        {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
        {"variable": ("sin_time_1D", ""), "edit": None},
        {"variable": ("cos_time_1D", ""), "edit": None},
        {"variable": ("Returns_Close", "EURUSD"), "edit": None},
        {"variable": ("Returns_Close", "USDJPY"), "edit": None},
        {"variable": ("Returns_Close", "EURJPY"), "edit": None},
        {"variable": ("Returns_Close", "GBPUSD"), "edit": None},
    ]

    tradable_markets = 'EURUSD'
    window_size = '3M'
    starting_balance = 10000
    look_back = 15
    # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001
    provision = 0.0001  # 0.001, cant be too high as it would not learn to trade

    # Training parameters
    batch_size = 8192  # 8192
    epochs = 10  # 40
    mini_batch_size = 256
    leverage = 10
    weight_decay = 0.00001
    l1_lambda = 1e-7
    num_episodes = 2000  # 100
    # Create the environment
    env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables,
                                    tradable_markets=tradable_markets, provision=provision,
                                    initial_balance=starting_balance, leverage=leverage,)
    agent = Transformer_PPO_Agent(n_actions=3,  # 3 actions: sell, hold, buy
                                  input_dims=env.calculate_input_dims(),
                                  gamma=0.9,
                                  alpha=0.0005,  # learning rate for actor network
                                  gae_lambda=0.9,  # lambda for generalized advantage estimation
                                  policy_clip=0.15,  # clip parameter for PPO
                                  entropy_coefficient=0.5,  # higher entropy coefficient encourages exploration
                                  batch_size=batch_size,
                                  n_epochs=epochs,
                                  mini_batch_size=mini_batch_size,
                                  weight_decay=weight_decay,
                                  l1_lambda=l1_lambda,
                                  static_input_dims=1)

    total_rewards = []
    episode_durations = []
    total_balances = []
    probs_dfs = {}
    balances_dfs = {}

    episode_probabilities = {'train': [], 'validation': [], 'test': []}

    # Split validation and test datasets into multiple rolling windows
    window_size_2 = '1M'
    test_rolling_datasets = rolling_window_datasets(df_test, window_size=window_size_2, look_back=look_back)
    val_rolling_datasets = rolling_window_datasets(df_validation, window_size=window_size_2, look_back=look_back)

    # Generate index labels for each rolling window dataset
    def generate_index_labels(rolling_datasets, dataset_type):
        index_labels = []
        for dataset in rolling_datasets:
            last_day = dataset.index[-1].strftime('%Y-%m-%d')
            label = f"{dataset_type}_{last_day}"
            index_labels.append(label)
        return index_labels

    val_labels = generate_index_labels(val_rolling_datasets, 'validation')
    test_labels = generate_index_labels(test_rolling_datasets, 'test')
    all_labels = val_labels + test_labels

    # Create a DataFrame to hold backtesting results for all rolling windows
    columns = ['Agent generation', 'Label', 'Final Balance', 'Number of Trades']
    backtest_results = pd.DataFrame(columns=columns)

    # Rolling DF
    rolling_datasets = rolling_window_datasets(df_train, window_size=window_size, look_back=look_back)
    dataset_iterator = cycle(rolling_datasets)
    generations_before = 0

    for episode in tqdm(range(num_episodes)):
        window_df = next(dataset_iterator)
        dataset_index = episode % len(rolling_datasets)
        print(f"Episode {episode + 1}: Learning from dataset with Start Date = {window_df.index.min()}, End Date = {window_df.index.max()}, len = {len(window_df)}")

        # Create a new environment with the randomly selected window's data
        env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables,
                                        tradable_markets=tradable_markets, provision=provision,
                                        initial_balance=starting_balance, leverage=leverage)

        observation = env.reset()
        done = False
        cumulative_reward = 0
        start_time = time.time()
        initial_balance = env.balance

        while not done:  # TODO check if this is correct
            current_position = env.current_position
            action, prob, val = agent.choose_action(observation, current_position)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, prob, val, reward, done, current_position)
            observation = observation_
            cumulative_reward += reward

            # Check if enough data is collected or if the dataset ends
            if len(agent.memory.states) >= agent.memory.batch_size:  # or done:
                agent.learn()
                agent.memory.clear_memory()

        if agent.generation > generations_before:
            start_time_2 = time.time()
            print("Backtesting on training, validation, and test datasets")

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for df, label in zip(val_rolling_datasets + test_rolling_datasets, val_labels + test_labels):
                    future = executor.submit(backtest_wrapper_AC, df, agent, 'EURUSD', look_back, variables, provision,starting_balance, leverage, Trading_Environment_Basic)
                    futures.append((future, label))

                # Process completed tasks and append results to DataFrame
                for future, label in futures:
                    balance, total_reward, number_of_trades, probs_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances = future.result()
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
                    temp_df = pd.DataFrame([result_data])
                    backtest_results = pd.concat([backtest_results, temp_df], ignore_index=True)

                    # Store probabilities and balances for plotting
                    probs_dfs[(agent.generation, label)] = probs_df
                    balances_dfs[(agent.generation, label)] = balances

            generations_before = agent.generation
            end_time_2 = time.time()
            episode_time_2 = end_time_2 - start_time_2
            print(f"Backtesting completed in {episode_time_2:.2f} seconds")

        # results
        end_time = time.time()
        episode_time = end_time - start_time
        total_rewards.append(cumulative_reward)
        episode_durations.append(episode_time)
        total_balances.append(env.balance)

        print(f"Completed learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
        print("-----------------------------------\n")

    # Plotting the results after all episodes
    backtest_results.set_index('Agent generation', inplace=True)
    print(backtest_results)

    from backtest.plots.generation_plot import plot_results, plot_total_rewards, plot_episode_durations, \
        plot_total_balances

    plot_results(backtest_results, ['Final Balance', 'Number of Trades', 'Total Reward'], agent.get_name())
    plot_total_rewards(total_rewards, agent.get_name())
    plot_episode_durations(episode_durations, agent.get_name())
    plot_total_balances(total_balances, agent.get_name())

    from backtest.plots.OHLC_probability_plot import PnL_generation_plot, Probability_generation_plot

    PnL_generation_plot(balances_dfs, port_number=8050)
    Probability_generation_plot(probs_dfs, port_number=8051)

    print('end')
