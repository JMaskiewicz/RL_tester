"""
DDQN 3.1

# TODO LIST
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- try transformer architecture or TFT transformer (Temporal Fusion Transformers transformer time series)

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

from data.function.load_data import load_data_parallel
from data.function.rolling_window import rolling_window_datasets
from data.function.edit import normalize_data, standardize_data
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
from functions.utilis import save_model
import backtest.backtest_functions.functions as BF

def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
    reward = 0
    reward_return = (current_close / previous_close - 1) * current_position * leverage
    if previous_position != current_position:
        provision_cost = provision * (abs(current_position) == 1)
    else:
        provision_cost = 0

    if reward_return < 0:
        reward_return = 2 * reward_return

    reward += reward_return * 10000 - provision_cost * 100

    return reward

def generate_predictions_and_backtest_DQN(df, agent, mkf, look_back, variables, provision=0.001, initial_balance=10000, leverage=1, Trading_Environment_Basic=None, plot=False):
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
    agent.q_policy.eval()

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

    # Switch back to training mode
    agent.q_policy.train()

    return balance, total_reward, number_of_trades, probabilities_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances

def backtest_wrapper_DQN(df, agent, mkf, look_back, variables, provision, initial_balance, leverage,
                        Trading_Environment_Basic=None):
    """
    # TODO add description
    DQN - Actor Critic
    """
    return generate_predictions_and_backtest_DQN(df, agent, mkf, look_back, variables, provision, initial_balance,
                                                leverage, Trading_Environment_Basic)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, dropout_rate=1/8):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 2048)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_value = nn.Linear(1024, 512)
        self.value = nn.Linear(512, 1)
        self.fc_advantage = nn.Linear(1024, 512)
        self.advantage = nn.Linear(512, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        val = F.relu(self.fc_value(x))
        val = self.value(val)

        adv = F.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        # Combine value and advantage streams
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))

        return q_values

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
    def __init__(self, input_dims, n_actions, epochs=1, mini_batch_size=256, gamma=0.99, policy_alpha=0.001, target_alpha=0.0005 , epsilon=1.0, epsilon_dec=1e-5, epsilon_end=0.01, mem_size=100000, batch_size=64, replace=1000, weight_decay=0.0005, l1_lambda=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda

        self.memory = ReplayBuffer(mem_size, (input_dims,), n_actions)
        self.q_policy = DuelingQNetwork(input_dims, n_actions).to(self.device)
        self.q_target = DuelingQNetwork(input_dims, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()

        self.policy_lr = policy_alpha
        self.target_lr = target_alpha
        self.policy_optimizer = optim.Adam(self.q_policy.parameters(), lr=self.policy_lr, weight_decay=weight_decay)
        self.target_optimizer = optim.Adam(self.q_target.parameters(), lr=self.target_lr, weight_decay=weight_decay)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        start_time = time.time()
        self.replace_target_network()

        # Set the policy network to training mode
        self.q_policy.train()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        for epoch in range(self.epochs):  # Loop over epochs
            num_mini_batches = max(self.batch_size // self.mini_batch_size, 1)

            for mini_batch in range(num_mini_batches):
                start = mini_batch * self.mini_batch_size
                end = min((mini_batch + 1) * self.mini_batch_size, self.batch_size)

                mini_states = torch.tensor(states[start:end], dtype=torch.float).to(self.device)
                mini_actions = torch.tensor(actions[start:end]).to(self.device)
                mini_rewards = torch.tensor(rewards[start:end], dtype=torch.float).to(self.device)
                mini_states_ = torch.tensor(states_[start:end], dtype=torch.float).to(self.device)
                mini_dones = torch.tensor(dones[start:end], dtype=torch.bool).to(self.device)

                self.policy_optimizer.zero_grad()

                q_pred = self.q_policy(mini_states).gather(1, mini_actions.unsqueeze(-1)).squeeze(-1)
                q_next = self.q_target(mini_states_).detach()
                q_eval = self.q_policy(mini_states_).detach()

                max_actions = torch.argmax(q_eval, dim=1)
                q_next[mini_dones] = 0.0
                q_target = mini_rewards + self.gamma * q_next.gather(1, max_actions.unsqueeze(-1)).squeeze(-1)

                # MSE loss
                loss = F.mse_loss(q_pred, q_target)

                # Calculate L1 penalty for all parameters
                l1_penalty = sum(p.abs().sum() for p in self.q_policy.parameters())
                total_loss = loss + self.l1_lambda * l1_penalty

                total_loss.backward()
                self.policy_optimizer.step()

        self.learn_step_counter += 1
        print('Epsilon decreasing...')
        self.decrement_epsilon()
        end_time = time.time()
        episode_time = end_time - start_time
        print(f"Learning of agent generation {self.generation} completed in {episode_time} seconds")



    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Convert the observation to a numpy array if it's not already
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation)
            # Reshape the observation to (1, -1) which means 1 row and as many columns as necessary
            observation = observation.reshape(1, -1)
            # Convert the numpy array to a tensor
            state = torch.tensor(observation, dtype=torch.float).to(self.device)
            actions = self.q_policy(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def get_action_q_values(self, observation, static_input):
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

    def choose_best_action(self, observation, static_input):
        """
        Selects the best action based on the highest Q-value without exploration.
        """
        q_values = self.get_action_q_values(observation)
        best_action = np.argmax(q_values)
        return best_action

    def get_epsilon(self):
        return self.epsilon

    def get_action_probabilities(self, observation, static_input):
        """
        Returns the probabilities of each action for a given observation.
        Converts Q values to a probability distribution using the softmax function.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        with torch.no_grad():
            q_values = self.q_policy(state)
            probabilities = F.softmax(q_values, dim=1).cpu().numpy()

        return probabilities.flatten()

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



