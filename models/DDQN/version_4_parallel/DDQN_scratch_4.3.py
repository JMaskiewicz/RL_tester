"""
DDQN 4.3

# TODO LIST
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- try transformer architecture or TFT transformer (Temporal Fusion Transformers transformer time series)

"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import multiprocessing
from multiprocessing import Process, Queue, Event, Manager
from threading import Thread
import sys
import time
from time import perf_counter, sleep
from functools import wraps
from typing import Callable, Any
from numba import jit
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import gym
from gym import spaces
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

from data.function.load_data import load_data_parallel
from data.function.rolling_window import rolling_window_datasets
from data.function.edit import process_variable
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
# import backtest.backtest_functions.functions as BF
from functions.utilis import save_actor_critic_model

"""
Reward Calculation function is the most crucial part of the RL algorithm. It is the function that determines the reward the agent receives for its actions.
"""
# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


@jit(nopython=True)
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
        reward *= 2  # penalty for wrong action

    # Calculate the cost of provision if the position has changed, and it's not neutral (0).
    if current_position != previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 - provision) * 1  # penalty for changing position
    elif current_position == previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 + provision) * 0  # small premium for holding position
    else:
        provision_cost = 0

    # Apply the provision cost
    reward += provision_cost

    # Scale the reward to enhance its significance for the learning process
    final_reward = reward * 100

    return final_reward


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dims, static_state_dims, n_actions, dropout_rate=1 / 8):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims + static_state_dims, 32)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_value = nn.Linear(32, 32)
        self.value = nn.Linear(32, 1)
        self.fc_advantage = nn.Linear(32, 32)
        self.advantage = nn.Linear(32, n_actions)

    def forward(self, state, static_state):
        x = torch.cat((state, static_state), dim=-1)
        print(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        print(x)
        val = F.relu(self.fc_value(x))
        val = self.value(val)

        adv = F.relu(self.fc_advantage(x))
        adv = self.advantage(adv)
        print(val, adv)
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
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.static_state_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, static_state):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.static_state_memory[index] = static_state
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        static_states = self.static_state_memory[batch]

        return states, actions, rewards, states_, dones, static_states

    def clear_memory(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros_like(self.state_memory)
        self.new_state_memory = np.zeros_like(self.new_state_memory)
        self.action_memory = np.zeros_like(self.action_memory)
        self.reward_memory = np.zeros_like(self.reward_memory)
        self.terminal_memory = np.zeros_like(self.terminal_memory)
        self.static_state_memory = np.zeros_like(self.static_state_memory)


class DDQN_Agent:
    def __init__(self, input_dims, n_actions, n_epochs=1, mini_batch_size=256, gamma=0.99, policy_alpha=0.001, target_alpha=0.0005, epsilon=1.0, epsilon_dec=1e-5, epsilon_end=0.01, mem_size=100000, batch_size=64, replace=1000, weight_decay=0.0005, l1_lambda=1e-5, static_input_dims=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.n_epochs = n_epochs  # Number of epochs
        self.mini_batch_size = mini_batch_size  # Mini-batch size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon   # Exploration rate
        self.epsilon_dec = epsilon_dec  # Exploration rate decay
        self.epsilon_min = epsilon_end  # Minimum exploration rate
        self.mem_size = mem_size  # Memory size
        self.batch_size = batch_size  # Batch size
        self.replace_target_cnt = replace  # Replace target network count
        self.learn_step_counter = 0  # Learn step counter
        self.weight_decay = weight_decay  # Weight decay
        self.l1_lambda = l1_lambda  # L1 regularization lambda
        self.action_space = [i for i in range(n_actions)]

        self.memory = ReplayBuffer(mem_size, (input_dims,), n_actions)
        self.q_policy = DuelingQNetwork(input_dims, static_input_dims, n_actions).to(self.device)
        self.q_target = DuelingQNetwork(input_dims, static_input_dims, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()  # Set the target network to evaluation mode

        self.policy_lr = policy_alpha  # Policy learning rate
        self.target_lr = target_alpha  # Target learning rate
        self.policy_optimizer = optim.Adam(self.q_policy.parameters(), lr=self.policy_lr, weight_decay=weight_decay)
        self.target_optimizer = optim.Adam(self.q_target.parameters(), lr=self.target_lr, weight_decay=weight_decay)

        # track the generation of the agent
        self.generation = 0

    def store_transition(self, state, action, reward, state_, done, static_state):
        self.memory.store_transition(state, action, reward, state_, done, static_state)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        # track the time it takes to learn
        start_time = time.time()
        print("-" * 100)
        self.replace_target_network()

        # Set the policy network to training mode
        self.q_policy.train()

        states, actions, rewards, states_, dones, static_states = self.memory.sample_buffer(self.batch_size)

        for epoch in range(self.n_epochs):  # Loop over epochs
            num_mini_batches = max(self.batch_size // self.mini_batch_size, 1)

            for mini_batch in range(num_mini_batches):
                start = mini_batch * self.mini_batch_size
                end = min((mini_batch + 1) * self.mini_batch_size, self.batch_size)

                mini_states = torch.tensor(states[start:end], dtype=torch.float).to(self.device)
                mini_actions = torch.tensor(actions[start:end]).to(self.device)
                mini_rewards = torch.tensor(rewards[start:end], dtype=torch.float).to(self.device)
                mini_states_ = torch.tensor(states_[start:end], dtype=torch.float).to(self.device)
                mini_dones = torch.tensor(dones[start:end], dtype=torch.bool).to(self.device)
                mini_static_states = torch.tensor(static_states[start:end], dtype=torch.float).to(self.device)
                mini_static_states = mini_static_states.unsqueeze(-1)

                self.policy_optimizer.zero_grad()

                q_pred = self.q_policy(mini_states, mini_static_states).gather(1, mini_actions.unsqueeze(-1)).squeeze(-1)
                q_next = self.q_target(mini_states_, mini_static_states).detach()
                q_eval = self.q_policy(mini_states_, mini_static_states).detach()

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

        self.decrement_epsilon()

        # Clear memory after learning
        self.memory.clear_memory()

        # Increment generation of the agent
        self.generation += 1

        # track the time it takes to learn
        end_time = time.time()
        episode_time = end_time - start_time

        # print the time it takes to learn
        print(f"\nLearning of agent generation {self.generation} completed in {episode_time} seconds")
        print("-" * 100)

    @torch.no_grad()
    def get_action_probabilities(self, observation, static_input):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        static_input = np.array([static_input])
        static_input_tensor = torch.tensor(static_input, dtype=torch.float).to(self.device)
        static_input_expanded = static_input_tensor.unsqueeze(0).expand(state.size(0), -1)

        q_values = self.q_policy(state, static_input_expanded)
        probabilities = F.softmax(q_values, dim=1).cpu().numpy()
        return probabilities.flatten()

    @torch.no_grad()
    def choose_action(self, observation, static_input):
        if np.random.random() > self.epsilon:
            # Prepare static input tensor
            static_input_tensor = torch.tensor([static_input], dtype=torch.float).to(self.device)
            observation_tensor = torch.tensor([observation], dtype=torch.float).to(self.device)
            # Call q_policy with both observation and static input
            actions = self.q_policy(observation_tensor, static_input_tensor)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    @torch.no_grad()
    def get_action_q_values(self, observation, static_input):
        """
        Returns the Q-values of each action for a given observation.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        # Prepare static input
        static_input_tensor = torch.tensor([static_input], dtype=torch.float).to(self.device)
        observation_tensor = torch.tensor([observation], dtype=torch.float).to(self.device)
        # Call q_policy with both observation and static input
        q_values = self.q_policy(observation_tensor, static_input_tensor)

        return q_values.cpu().numpy()

    @torch.no_grad()
    def choose_best_action(self, observation, static_input):
        """
        Selects the best action based on the highest Q-value without exploration.
        """
        q_values = self.get_action_q_values(observation, static_input)
        best_action = np.argmax(q_values)
        return best_action

    def get_epsilon(self):
        return self.epsilon

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
def get_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = perf_counter()

        print(f'"{func.__name__}()" took {end_time - start_time:.3f} seconds to execute')
        return result

    return wrapper

def print_signal_status(arg1=None, **kwargs):
    def print_status(signals):
        print(f"Status of Signals:")
        for key, value in signals.items():
            if isinstance(value, list):
                for i, signal in enumerate(value, start=1):
                    status = 'Set' if signal.is_set() else 'Not Set'
                    print(f"{key} {i}: {status}")
            else:
                status = 'Set' if value.is_set() else 'Not Set'
                print(f"{key}: {status}")

    if callable(arg1):
        func = arg1
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self = args[0]

            print(f"Status of Signals after {func.__name__}:")
            signals = {
                'Work Event': self.work_event,
                'Pause Signals': self.pause_signals,
                'Resume Signals': self.resume_signals,
                'Backtesting Completed': self.backtesting_completed,
            }
            print_status(signals)
            return result

        return wrapper
    else:
        print_status(arg1 or kwargs)

def progress_update(interval=60):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Dynamically retrieve arguments based on function signature
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            arg_dict = dict(zip(arg_names, args))
            arg_dict.update(kwargs)

            # Ensure required arguments are present
            required_args = ['shared_episodes_counter', 'max_episodes_per_worker', 'num_workers']
            missing_args = [arg for arg in required_args if arg not in arg_dict]
            if missing_args:
                raise ValueError(f"Missing required argument(s): {', '.join(missing_args)}")

            def progress_updater(stop_event, shared_episodes_counter, max_episodes_per_worker, num_workers):
                """Inner function to update the progress based on shared counter and total episodes."""
                start_time = time.time()
                while not stop_event.is_set():
                    current_episodes = shared_episodes_counter.value
                    total_episodes = max_episodes_per_worker * num_workers
                    progress_percentage = (current_episodes / total_episodes) * 100
                    bar_length = 50
                    filled_length = int(round(bar_length * progress_percentage / 100))
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    elapsed_time = time.time() - start_time
                    speed = current_episodes / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = ((total_episodes - current_episodes) / speed) if speed > 0 else 0
                    formatted_elapsed_time = format_time(elapsed_time)
                    formatted_eta = format_time(eta_seconds)
                    sys.stdout.write(
                        f"\033[92m\rProgress: {progress_percentage:.0f}% |{bar}| {current_episodes}/{total_episodes} [Elapsed: {formatted_elapsed_time}, ETA: {formatted_eta}, Speed: {speed:.2f}it/s]\033[0m")
                    sys.stdout.flush()
                    time.sleep(interval)

            # Start the progress updater thread
            stop_event = Event()
            updater_thread = Thread(target=progress_updater, args=(
                stop_event,
                arg_dict['shared_episodes_counter'],
                arg_dict['max_episodes_per_worker'],
                arg_dict['num_workers']
            ))
            updater_thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                updater_thread.join()
            return result

        return wrapper
    return decorator

def generate_predictions_and_backtest(agent_type, df, agent, mkf, look_back, variables, provision=0.001, starting_balance=10000, leverage=1, Trading_Environment_Basic=None):
    """
    # TODO add description
    """
    action_probabilities_list = []
    best_action_list = []
    balances = []
    number_of_trades = 0

    # Preparing the environment
    eval_methods = [('actor', 'critic'), ('q_policy',)]
    for method_group in eval_methods:
        for method in method_group:
            if hasattr(agent, method):
                getattr(agent, method).eval()

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
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(df)-env.look_back) if returns.std() > 1e-6 else float('nan')

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    negative_volatility = returns[returns < 0].std() * np.sqrt(len(df)-env.look_back)
    sortino_ratio = returns.mean() / negative_volatility if negative_volatility > 1e-6 else float('nan')

    annual_return = cumulative_returns.iloc[-1] ** ((len(df)-env.look_back) / len(returns)) - 1
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else float('nan')

    # Convert the list of action probabilities to a DataFrame
    probabilities_df = pd.DataFrame(action_probabilities_list, columns=['Short', 'Neutral', 'Long'])
    action_df = pd.DataFrame(best_action_list, columns=['Action'])

    # Ensure the agent's networks are reverted back to training mode
    for method_group in eval_methods:
        for method in method_group:
            if hasattr(agent, method):
                getattr(agent, method).train()

    return env.balance, env.reward_sum, number_of_trades, probabilities_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances

def backtest_wrapper_AC(agent_type, df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic=None):
    """
    # TODO add description
    """
    return generate_predictions_and_backtest(agent_type, df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic)

@get_time
#@print_signal_status
def backtest_in_background(agent_type, agent, backtest_results, num_workers_backtesting, val_rolling_datasets, test_rolling_datasets, val_labels, test_labels, probs_dfs, balances_dfs, backtesting_completed):
    start_time = time.time()
    print("Starting backtesting...")
    with ThreadPoolExecutor(max_workers=num_workers_backtesting) as executor:
        futures = []
        for df, label in zip(val_rolling_datasets + test_rolling_datasets, val_labels + test_labels):
            future = executor.submit(backtest_wrapper_AC, agent_type, df, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage, Trading_Environment_Basic)
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


def environment_worker(agent_type, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signal, resume_signal, total_rewards, total_balances, worker_id, individual_worker_batch_size, workers_completed, workers_completed_signal, shared_episodes_counter):
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
            # TODO here
            action = agent.choose_action(observation, env.current_position)
            observation_, reward, done, info = env.step(action)
            experience = (observation, action, reward, observation_, done, env.current_position)
            shared_queue.put(experience)
            observation = observation_

            experiences_collected += 1  # Increment the counter for each experience collected
            if experiences_collected >= individual_worker_batch_size:
                end_time = time.time()  # Record the time when the batch size limit is reached
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                print(f"Worker {multiprocessing.current_process().name}: Reached individual batch size limit in {elapsed_time:.2f} seconds.")
                pause_signal.set()
                # print_signal_status({'Pause Signal': pause_signal, 'backtesting_completed': workers_completed_signal, 'Work Event': work_event, 'Resume Signal': resume_signal})
                while pause_signal.is_set():
                    time.sleep(0.1)

            if workers_completed_signal.is_set():
                break

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

@get_time
def collect_and_learn(agent_type, dfs, max_episodes_per_worker, env_settings, batch_size_for_learning, backtest_results, agent,
                      num_workers, num_workers_backtesting, backtesting_frequency=1):

    manager = Manager()
    (total_rewards, total_balances, shared_episodes_counter, workers_completed, backtesting_completed,
     work_event, pause_signals, resume_signals, workers_completed_signal,
     shared_queue) = setup_shared_resources_and_events(manager, num_workers)

    workers = start_workers(agent_type, num_workers, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event,
                            pause_signals, resume_signals, total_rewards, total_balances, workers_completed,
                            workers_completed_signal, shared_episodes_counter, batch_size_for_learning)

    manage_learning_and_backtesting(agent_type, agent, num_workers_backtesting, backtest_results, backtesting_completed, work_event,
                                    pause_signals, resume_signals, shared_queue, workers_completed_signal,
                                    shared_episodes_counter, total_rewards, total_balances, batch_size_for_learning,
                                    backtesting_frequency, max_episodes_per_worker, num_workers)

    for worker in workers:
        worker.join()

    print("\r" + " " * 100, end='')
    print("All workers stopped.")
    return list(total_rewards), list(total_balances)

@get_time
@progress_update(interval=10)
def manage_learning_and_backtesting(agent_type, agent, num_workers_backtesting, backtest_results, backtesting_completed, work_event, pause_signals, resume_signals, shared_queue, workers_completed_signal, shared_episodes_counter, total_rewards, total_balances, batch_size_for_learning, backtesting_frequency, max_episodes_per_worker=10, num_workers=4):
    agent_generation = 0
    try:
        total_experiences = 0
        while True:
            if not work_event.is_set():
                work_event.set()

            while not shared_queue.empty():
                experience = shared_queue.get_nowait()
                agent.store_transition(*experience)
                total_experiences += 1

            if total_experiences >= batch_size_for_learning and backtesting_completed.is_set():
                print("\nLearning phase initiated.")
                agent.learn()
                total_experiences = 0
                agent.memory.clear_memory()

                for signal in pause_signals:
                    signal.clear()

                work_event.set()

                for resume_signal in resume_signals:
                    resume_signal.set()

                if agent.generation > agent_generation and agent.generation % backtesting_frequency == 0:
                    backtesting_completed.clear()
                    backtest_thread = Thread(target=backtest_in_background, args=(agent_type, agent, backtest_results, num_workers_backtesting, val_rolling_datasets, test_rolling_datasets, val_labels, test_labels, probs_dfs, balances_dfs, backtesting_completed))
                    backtest_thread.start()
                    agent_generation = agent.generation

            if workers_completed_signal.is_set():
                backtesting_completed.wait()
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    print("All workers stopped.")
    return None

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

def start_workers(agent_type, num_workers, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signals, resume_signals, total_rewards, total_balances, workers_completed, workers_completed_signal, shared_episodes_counter, batch_size_for_learning):
    workers = []
    for i in range(num_workers):
        worker_process = Process(target=environment_worker, args=(agent_type, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signals[i], resume_signals[i], total_rewards, total_balances, i+1, batch_size_for_learning // num_workers, workers_completed, workers_completed_signal, shared_episodes_counter))
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

    add_time_sine_cosine(df, '1W')
    df[("sin_time_1W", "")] = df[("sin_time_1W", "")] / 2 + 0.5
    df[("cos_time_1W", "")] = df[("cos_time_1W", "")] / 2 + 0.5
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
        #{"variable": ("Close", "EURJPY"), "edit": "standardize"},
        #{"variable": ("Close", "GBPUSD"), "edit": "standardize"},
        #{"variable": ("RSI_14", "EURUSD"), "edit": "standardize"},
        #{"variable": ("ATR_24", "EURUSD"), "edit": "standardize"},
        #{"variable": ("sin_time_1W", ""), "edit": None},
        #{"variable": ("cos_time_1W", ""), "edit": None},
        #{"variable": ("Returns_Close", "EURUSD"), "edit": None},
        #{"variable": ("Returns_Close", "USDJPY"), "edit": None},
        #{"variable": ("Returns_Close", "EURJPY"), "edit": None},
        #{"variable": ("Returns_Close", "GBPUSD"), "edit": None},
    ]

    tradable_markets = 'EURUSD'
    window_size = '1Y'
    starting_balance = 10000
    look_back = 20  # 20
    # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001
    provision = 0.0001  # 0.001, cant be too high as it would not learn to trade

    # Training parameters
    batch_size = 1024  # 8192
    epochs = 1  # 40
    mini_batch_size = 128  # 256
    leverage = 10  # 30
    l1_lambda = 1e-7  # L1 regularization
    weight_decay = 0.000001  # L2 regularization

    # Number of CPU cores for number of workers
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    num_workers = min(max(1, num_cores - 1), 4)  # Number of workers, some needs to left for backtesting
    num_workers_backtesting = 8  # backtesting is parallelized in same time that gathering data for next generation
    num_episodes = 100  # need to be divisible by num_workers
    max_episodes_per_worker = num_episodes // num_workers

    '''
    Number of workers need to be related to window sizes to have best performance
    Backtesting is bottleneck now maybe its better to backtest only each 5th generation
    '''

    # Split validation and test datasets into multiple rolling windows
    # TODO add last year of training data to validation set
    window_size_2 = '3M'
    test_rolling_datasets = rolling_window_datasets(df_test, window_size=window_size_2, look_back=look_back)
    val_rolling_datasets = rolling_window_datasets(df_validation, window_size=window_size_2, look_back=look_back)

    # Generate index labels for each rolling window dataset
    val_labels = generate_index_labels(val_rolling_datasets, 'validation')
    test_labels = generate_index_labels(test_rolling_datasets, 'test')
    all_labels = val_labels + test_labels

    # Create a DataFrame to hold backtesting results for all rolling windows
    backtest_results = {}

    # Create an instance of the agent
    agent = DDQN_Agent(input_dims=len(variables) * look_back,
                       n_actions=3,
                       n_epochs=epochs,
                       mini_batch_size=mini_batch_size,
                       policy_alpha=0.0005,
                       target_alpha=0.0005,
                       gamma=0.9,
                       epsilon=1.0,
                       epsilon_dec=0.99,
                       epsilon_end=0,
                       mem_size=100000,
                       batch_size=batch_size,
                       replace=5,
                       weight_decay=weight_decay,
                       l1_lambda=l1_lambda,
                       static_input_dims=1,
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
    total_rewards, total_balances = collect_and_learn('DQN', rolling_datasets, max_episodes_per_worker, env_settings, batch_size, backtest_results, agent, num_workers, num_workers_backtesting, backtesting_frequency=5)
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
