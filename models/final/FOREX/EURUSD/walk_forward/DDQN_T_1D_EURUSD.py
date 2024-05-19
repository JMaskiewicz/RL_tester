"""
DDQN 5.1

- added TD lambda learning
- added alternative rewards


# TODO add saving the model

"""
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import random
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from numba import jit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

"""
Reward Calculation function is the most crucial part of the RL algorithm. It is the function that determines the reward the agent receives for its actions.
currently there is 
"""

@jit(nopython=True)
def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
    # Calculate the normal return
    if previous_close != 0:
        normal_return = (current_close - previous_close) / previous_close
    else:
        normal_return = 0

    # Calculate the base reward
    reward = normal_return * current_position * 1000

    # Penalize the agent for taking the wrong action
    if reward < 0:
        reward *= 1  # penalty for wrong action

    # Calculate the cost of provision if the position has changed, and it's not neutral (0).
    if current_position != previous_position and abs(current_position) == 1:
        provision_cost = - provision * 1000  # penalty for changing position
    elif current_position == previous_position and abs(current_position) == 1:
        provision_cost = + provision * 0
    else:
        provision_cost = 0

    # Apply the provision cost
    reward += provision_cost

    # Scale the reward to enhance its significance for the learning process
    final_reward = reward

    return final_reward

class TransformerDuelingQNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, n_heads=4, n_layers=2, dropout_rate=1/4, static_input_dims=1):
        super(TransformerDuelingQNetwork, self).__init__()
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

        self.fc1 = nn.Linear(input_dims * 2, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc_value = nn.Linear(512, 1)
        self.fc_advantage = nn.Linear(512, n_actions)

    def forward(self, dynamic_state, static_state):
        if len(dynamic_state.shape) == 2:
            dynamic_state = dynamic_state.unsqueeze(1)  # Adding sequence dimension if it's missing

        batch_size, seq_length, _ = dynamic_state.size()
        positional_encoding = self.positional_encoding[:, :seq_length, :].expand(batch_size, -1, -1)

        dynamic_state = dynamic_state + positional_encoding
        transformer_out = self.transformer_encoder(dynamic_state)

        static_state_encoded = self.fc_static(static_state.unsqueeze(1))
        combined_features = torch.cat((transformer_out[:, -1, :], static_state_encoded.squeeze(1)), dim=1)

        x = torch.relu(self.fc1(combined_features))
        x = torch.relu(self.fc2(x))

        val = self.fc_value(x)
        adv = self.fc_advantage(x)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))

        return q_values


class DQNMemory:
    def __init__(self, max_size, dynamic_input_shape, static_input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Memory arrays for dynamic (time-series) state components
        self.dynamic_state_memory = np.zeros((self.mem_size, *dynamic_input_shape), dtype=np.float32)
        self.dynamic_new_state_memory = np.zeros((self.mem_size, *dynamic_input_shape), dtype=np.float32)

        # Memory array for static state components
        self.static_state_memory = np.zeros((self.mem_size, *static_input_shape), dtype=np.float32)
        self.static_new_state_memory = np.zeros((self.mem_size, *static_input_shape), dtype=np.float32)

        # Memory for other components
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, dynamic_state, static_state, action, reward, dynamic_state_, static_state_, done):
        index = self.mem_cntr % self.mem_size
        self.dynamic_state_memory[index] = dynamic_state
        self.dynamic_new_state_memory[index] = dynamic_state_
        self.static_state_memory[index] = static_state
        self.static_new_state_memory[index] = static_state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        dynamic_states = self.dynamic_state_memory[batch]
        dynamic_states_ = self.dynamic_new_state_memory[batch]
        static_states = self.static_state_memory[batch]
        static_states_ = self.static_new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return dynamic_states, static_states, actions, rewards, dynamic_states_, static_states_, dones

    def clear_memory(self):
        self.mem_cntr = 0
        self.dynamic_state_memory = np.zeros_like(self.dynamic_state_memory)
        self.dynamic_new_state_memory = np.zeros_like(self.dynamic_new_state_memory)
        self.static_state_memory = np.zeros_like(self.static_state_memory)
        self.static_new_state_memory = np.zeros_like(self.static_new_state_memory)
        self.action_memory = np.zeros_like(self.action_memory)
        self.reward_memory = np.zeros_like(self.reward_memory)
        self.terminal_memory = np.zeros_like(self.terminal_memory)


class DDQN_Agent_T_1D_EURUSD:
    def __init__(self, input_dims, n_actions, n_epochs=1, mini_batch_size=256, gamma=0.99, policy_alpha=0.001,
                 target_alpha=0.0005, epsilon=1.0, epsilon_dec=1e-5, epsilon_end=0.01, mem_size=100000,
                 batch_size=64, replace=1000, weight_decay=0.0005, l1_lambda=1e-5, static_input_dims=1,
                 lr_decay_rate=0.999, premium_gamma=0.5, lambda_=0.75):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        print(f"Using device: {self.device}")
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.lr_decay_rate = lr_decay_rate
        self.premium_gamma = premium_gamma
        self.action_space = [i for i in range(n_actions)]
        self.lambda_ = lambda_

        dynamic_input_shape = (input_dims,)
        static_input_shape = (static_input_dims,)

        # Initialize memory with dynamic and static input shapes
        self.memory = DQNMemory(mem_size, dynamic_input_shape, static_input_shape, n_actions)

        # Initialize networks
        self.q_policy = TransformerDuelingQNetwork(n_actions, input_dims, n_heads=4, n_layers=2, dropout_rate=0.25, static_input_dims=static_input_dims).to(self.device)
        self.q_target = TransformerDuelingQNetwork(n_actions, input_dims, n_heads=4, n_layers=2, dropout_rate=0.25, static_input_dims=static_input_dims).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()

        self.policy_lr = policy_alpha
        self.target_lr = target_alpha
        self.policy_optimizer = optim.Adam(self.q_policy.parameters(), lr=self.policy_lr, weight_decay=weight_decay)
        self.target_optimizer = optim.Adam(self.q_target.parameters(), lr=self.target_lr, weight_decay=weight_decay)

        self.policy_scheduler = ExponentialLR(self.policy_optimizer, gamma=self.lr_decay_rate)
        self.target_scheduler = ExponentialLR(self.target_optimizer, gamma=self.lr_decay_rate)

        self.generation = 0

    def store_transition(self, dynamic_state, static_state, action, reward, dynamic_state_, static_state_, done):
        self.memory.store_transition(dynamic_state, static_state, action, reward, dynamic_state_, static_state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def learn(self):
        # track the time it takes to learn
        start_time = time.time()
        print('\n', "-" * 100)
        self.replace_target_network()

        # Set the policy network to training mode
        self.q_policy.train()

        dynamic_states, static_states, actions, rewards, dynamic_states_, static_states_, dones = self.memory.sample_buffer(self.batch_size)

        # Initialize eligibility traces
        eligibility_traces = torch.zeros((self.batch_size, self.n_actions)).to(self.device)

        for epoch in range(self.n_epochs):  # Loop over epochs
            # Calculate the number of mini-batches
            num_mini_batches = max(self.batch_size // self.mini_batch_size, 1)

            for mini_batch in range(num_mini_batches):
                # Calculate the start and end indices of the mini-batch
                start = mini_batch * self.mini_batch_size
                end = min((mini_batch + 1) * self.mini_batch_size, self.batch_size)

                # Convert the mini-batch to tensors
                mini_dynamic_states = torch.tensor(dynamic_states[start:end], dtype=torch.float).unsqueeze(1).to(self.device)
                mini_static_states = torch.tensor(static_states[start:end], dtype=torch.float).to(self.device)
                mini_dynamic_states_ = torch.tensor(dynamic_states_[start:end], dtype=torch.float).unsqueeze(1).to(self.device)
                mini_static_states_ = torch.tensor(static_states_[start:end], dtype=torch.float).to(self.device)
                mini_actions = torch.tensor(actions[start:end]).to(self.device)
                mini_rewards = torch.tensor(rewards[start:end], dtype=torch.float).to(self.device)
                mini_dones = torch.tensor(dones[start:end], dtype=torch.bool).to(self.device)

                q_pred = self.q_policy(mini_dynamic_states, mini_static_states).gather(1, mini_actions.unsqueeze(
                    -1)).squeeze(-1)

                # Predict Q-values for the next mini-batch states using target network
                q_next = self.q_target(mini_dynamic_states_, mini_static_states_).detach()
                q_eval = self.q_policy(mini_dynamic_states_, mini_static_states_).detach()

                max_actions = torch.argmax(q_eval, dim=1)
                q_next[mini_dones] = 0.0
                q_target = mini_rewards + self.gamma * q_next.gather(1, max_actions.unsqueeze(-1)).squeeze(-1)

                td_error = q_target - q_pred
                eligibility_traces_mini = eligibility_traces[start:end]  # work with the correct segment
                eligibility_traces_mini = self.gamma * self.lambda_ * eligibility_traces_mini
                eligibility_traces_mini.scatter_(1, mini_actions.unsqueeze(-1), 1)

                expanded_td_error = td_error.unsqueeze(1).expand_as(eligibility_traces_mini)
                loss = (expanded_td_error.pow(2) * eligibility_traces_mini).mean()

                l1_penalty = sum(p.abs().sum() for p in self.q_policy.parameters())
                total_loss = loss + self.l1_lambda * l1_penalty

                self.policy_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()

        # Decay learning rate
        self.policy_scheduler.step()
        self.target_scheduler.step()

        self.learn_step_counter += 1  # Increment the learn step counter
        self.decrement_epsilon()  # Decrement the exploration rate

        # Clear memory after learning
        self.memory.clear_memory()

        # Increment generation of the agent
        self.generation += 1

        # track the time it takes to learn
        end_time = time.time()
        episode_time = end_time - start_time

        # print the time it takes to learn
        print(f"Learning of agent generation {self.generation} completed in {episode_time} seconds")
        print("-" * 100)

    @torch.no_grad()
    def choose_action(self, dynamic_state, static_state):
        # Epsilon-greedy action selection
        if np.random.random() > self.epsilon:
            # Ensure both parts of the state are correctly formatted as tensors
            dynamic_state = torch.tensor(dynamic_state, dtype=torch.float).unsqueeze(0).to(
                self.device)  # Add batch dimension
            static_state = torch.tensor([static_state], dtype=torch.float).unsqueeze(0).to(
                self.device)  # Add batch dimension and wrap in list if it's a single value

            # Get Q-values from the policy network
            q_values = self.q_policy(dynamic_state, static_state)
            action = torch.argmax(q_values).item()  # Choose the action with the highest Q-value

        else:
            # If the random number is less than epsilon, take a random action
            action = np.random.choice(self.action_space)

        return action

    @torch.no_grad()
    def get_action_q_values(self, observation, current_position):
        """
        Returns the Q-values of each action for a given observation.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        dynamic_state = observation.reshape(1, -1)
        static_state = np.array([current_position]).reshape(1, -1)

        dynamic_state_tensor = torch.tensor(dynamic_state, dtype=torch.float).to(self.device)
        static_state_tensor = torch.tensor(static_state, dtype=torch.float).to(self.device)

        q_values = self.q_policy(dynamic_state_tensor, static_state_tensor)

        return q_values.cpu().numpy()

    @torch.no_grad()
    def choose_best_action(self, observation, current_position):
        """
        Selects the best action based on the highest Q-value without exploration.
        """
        q_values = self.get_action_q_values(observation, current_position)
        best_action = np.argmax(q_values)
        return best_action

    def get_epsilon(self):
        return self.epsilon

    @torch.no_grad()
    def get_action_probabilities(self, observation, current_position):
        """
        Returns the probabilities of each action for a given observation.
        Converts Q values to a probability distribution using the softmax function.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # Separate handling of dynamic and static states
        dynamic_state = observation.reshape(1, -1)
        static_state = np.array([current_position]).reshape(1, -1)  # Static state, reshaped for batch processing

        # Convert numpy arrays to tensors and ensure they are on the correct device
        dynamic_state_tensor = torch.tensor(dynamic_state, dtype=torch.float).to(self.device)
        static_state_tensor = torch.tensor(static_state, dtype=torch.float).to(self.device)

        # Pass both state components to the network
        q_values = self.q_policy(dynamic_state_tensor, static_state_tensor)

        # Compute the softmax to get action probabilities
        probabilities = F.softmax(q_values, dim=1).cpu().numpy()

        return probabilities.flatten()

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

    start_date = '2005-01-01'  # worth to keep 2008 as it was a financial crisis
    validation_date = '2017-12-31'
    test_date = '2019-01-01'
    end_date = '2020-01-01'
    test_date_2 = test_date

    final_test_results = pd.DataFrame()
    final_balance = 10000

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

    variables = [
        {"variable": ("Close", "USDJPY"), "edit": "standardize"},
        {"variable": ("Close", "EURUSD"), "edit": "standardize"},
        {"variable": ("Close", "EURJPY"), "edit": "standardize"},
        {"variable": ("Close", "GBPUSD"), "edit": "standardize"},
        {"variable": ("RSI_14", "EURUSD"), "edit": "standardize"},
        {"variable": ("ATR_24", "EURUSD"), "edit": "standardize"},
        # {"variable": ("sin_time_1W", ""), "edit": None},
        # {"variable": ("cos_time_1W", ""), "edit": None},
        {"variable": ("Returns_Close", "EURUSD"), "edit": None},
        {"variable": ("Returns_Close", "USDJPY"), "edit": None},
        {"variable": ("Returns_Close", "EURJPY"), "edit": None},
        {"variable": ("Returns_Close", "GBPUSD"), "edit": None},
    ]

    look_back = 20
    tradable_markets = 'EURUSD'
    # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001
    provision = 0.0001  # 0.001, cant be too high as it would not learn to trade
    leverage = 1

    # tracking
    provision_sum_test_final = 0

    for move_forward in range(1, 6):
        print("validation_date", validation_date)
        print("test_date", test_date)
        print("end_date", end_date)

        df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD'], '1D')

        add_indicators(df, indicators)
        add_returns(df, return_indicators)

        add_time_sine_cosine(df, '1W')

        df = df.dropna()
        df_train, df_validation, df_test = df[start_date:validation_date], df[validation_date:test_date], df[test_date:end_date]

        df_validation = pd.concat([df_train.iloc[-look_back:], df_validation])
        df_test = pd.concat([df_validation.iloc[-look_back:], df_test])

        window_size = '1Y'
        starting_balance = final_balance
        # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001

        # Environment parameters
        num_episodes = 5000  # 100

        # Instantiate the agent
        agent = DDQN_Agent_T_1D_EURUSD(input_dims=len(variables) * look_back,  # input dimensions
                                       n_actions=3,  # buy, sell, hold
                                       n_epochs=1,  # number of epochs 10
                                       mini_batch_size=64,  # mini batch size 128
                                       policy_alpha=0.0005,  # learning rate for the policy network  0.0005
                                       target_alpha=0.00005,  # learning rate for the target network
                                       gamma=0.75,  # discount factor 0.99
                                       epsilon=1.0,  # initial epsilon 1.0
                                       epsilon_dec=0.998,  # epsilon decay rate 0.99
                                       epsilon_end=0,  # minimum epsilon  0
                                       mem_size=1000000,   # memory size 100000
                                       batch_size=1024,  # batch size  1024
                                       replace=10,  # replace target network count 10
                                       weight_decay=0.000005,  # Weight decay
                                       l1_lambda=0.00000005,  # L1 regularization lambda
                                       lr_decay_rate=0.995,   # Learning rate decay rate
                                       premium_gamma=0.5,  # Discount factor for the alternative rewards
                                       lambda_=0.5,  # Lambda for TD(lambda) learning
                                       )

        total_rewards, episode_durations, total_balances = [], [], []
        episode_probabilities = {'train': [], 'validation': [], 'test': []}

        index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
        columns = ['Final Balance', 'Dataset Index']
        backtest_results = pd.DataFrame(index=index, columns=columns)

        window_size_2 = '2Y'
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

            dynamic_state = env.reset()
            done = False
            initial_balance = env.balance

            while not done:
                action = agent.choose_action(dynamic_state, env.current_position)
                dynamic_state_, reward, done, info = env.step(action)

                agent.store_transition(dynamic_state, env.current_position, action, reward, dynamic_state_,
                                       env.current_position, done)
                dynamic_state = dynamic_state_

                # Learning process
                if agent.memory.mem_cntr >= agent.batch_size:
                    agent.learn()
                    agent.memory.clear_memory()

                if generation < agent.generation:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = []
                        # Combine validation and test datasets and labels for processing
                        for df, label in zip(val_rolling_datasets + test_rolling_datasets, val_labels + test_labels):
                            future = executor.submit(
                                BF.backtest_wrapper, 'DQN', df, agent, tradable_markets, look_back,
                                variables, provision, starting_balance, leverage,
                                Trading_Environment_Basic, reward_calculation
                            )
                            futures.append((future, label))

                        # Process futures as they complete
                        for future, label in futures:
                            result = future.result()
                            # Make sure to update these variables to match the exact structure of the data being returned
                            balance, total_reward, number_of_trades, probs_df, action_df, sharpe_ratio, max_drawdown, \
                                sortino_ratio, calmar_ratio, cumulative_returns, balances, provision_sum, max_drawdown_duration, \
                                average_trade_duration, in_long, in_short, in_out_of_market = result

                            # Populate a dictionary with the results
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
                                'Calmar Ratio': calmar_ratio,
                                'Max Drawdown Duration': max_drawdown_duration,
                                'Average Trade Duration': average_trade_duration,
                                'In Long': in_long,
                                'In Short': in_short,
                                'In Out of the Market': in_out_of_market,
                            }

                            key = (agent.generation, label)
                            backtest_results.setdefault(key, []).append(result_data)
                            probs_dfs[key] = probs_df
                            balances_dfs[key] = balances

                        generation = agent.generation
                        print(f"Backtesting completed for {agent.get_name()} generation {generation}")

                # results
            end_time = time.time()
            episode_time = end_time - start_time
            total_rewards.append(env.reward_sum)
            episode_durations.append(episode_time)
            total_balances.append(env.balance)

            print(
                f"Completed learning fro selected window in episode {episode + 1}: Total Reward: {env.reward_sum}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds, Agent Epsilon: {agent.get_epsilon():.4f}")

        buy_and_hold_agent = Buy_and_hold_Agent()
        sell_and_hold_agent = Sell_and_hold_Agent()
        # TODO
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

        sah_results_prepared = sah_results_prepared.drop(('', 'Agent Generation'),
                                                         axis=1)  # drop the agent generation column
        bah_results_prepared = bah_results_prepared.drop(('', 'Agent Generation'),
                                                         axis=1)  # drop the agent generation column

        print('buy and hold final balance', bah_results_prepared[('BAH', 'Final Balance')][1])
        print('sell and hold final balance', sah_results_prepared[('SAH', 'Final Balance')][1])

        # Merge BAH and SAH results on 'Label'
        new_backtest_results = pd.merge(bah_results_prepared, sah_results_prepared, on=[('', 'Label')], how='outer')

        backtest_results = prepare_backtest_results(backtest_results, agent.get_name())
        backtest_results = pd.merge(backtest_results, new_backtest_results, on=[('', 'Label')], how='outer')
        backtest_results = backtest_results.set_index([('', 'Agent Generation')])

        # Find the index of the maximum Sharpe Ratio in the validation set
        label_series = backtest_results[('', 'Label')]
        backtest_results = backtest_results.drop(('', 'Label'), axis=1)
        backtest_results['Label'] = label_series

        # Filter rows where Label is 'validation'
        validation_set = backtest_results[backtest_results['Label'] == val_labels[0]]

        # Extract the Sharpe Ratio column for the validation set
        sharpe_ratios_validation = validation_set[(agent.get_name(), 'Sharpe Ratio')]

        # Find the index of the maximum Sharpe Ratio in the validation set
        best_sharpe_index = sharpe_ratios_validation.idxmax()

        # if nan in the sharpe ratios, take the last one
        if pd.isna(best_sharpe_index):
            best_sharpe_index = agent.generation - 1

        # Extract the best result row
        best_result = backtest_results.loc[best_sharpe_index]

        # Convert the result to a dictionary
        best_result_dict = best_result.to_dict()

        # Extract the corresponding balances for the best result
        agent_generation = best_sharpe_index
        balances_key = (agent_generation, test_labels[0])
        best_balances = balances_dfs.get(balances_key, [])

        probs_key = (agent_generation, test_labels[0])
        best_probs_df = probs_dfs.get(probs_key, pd.DataFrame())
        best_probs_df[f'{agent.get_name()}_Action'] = best_probs_df.idxmax(axis=1)
        best_balances_df = pd.DataFrame(best_balances, columns=[f'{agent.get_name()}_Balances'])

        dates = df_test.iloc[look_back:-1].index
        best_balances_df['Date'] = dates
        best_probs_df['Date'] = dates
        close_prices = df_test['Close', tradable_markets].reset_index()
        close_prices = close_prices.iloc[look_back:-1]
        close_prices.columns = ['Date', f'Close_{tradable_markets}']

        best_balances_df = pd.merge(best_balances_df, close_prices, on='Date', how='outer')
        best_balances_df = pd.merge(best_balances_df, best_probs_df, on='Date', how='outer')
        final_test_results = pd.concat([final_test_results, best_balances_df])

        final_balance = final_test_results[f'{agent.get_name()}_Balances'].iloc[-1]
        print(f"Final Balance: {final_balance:.2f}")

        # save also number of trades and provision sum
        provision_sum_test_final = provision_sum_test_final + \
                                   best_result_dict.get((agent.get_name(), 'Provision Sum'), 0)[agent_generation]

        # add year to the dates
        validation_date = (datetime.strptime(validation_date, '%Y-%m-%d') + relativedelta(years=1)).strftime('%Y-%m-%d')
        test_date = (datetime.strptime(test_date, '%Y-%m-%d') + relativedelta(years=1)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(end_date, '%Y-%m-%d') + relativedelta(years=1)).strftime('%Y-%m-%d')

        # final results for the agent

    df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD'], '1D')

    add_indicators(df, indicators)
    add_returns(df, return_indicators)

    add_time_sine_cosine(df, '1W')

    df = df.dropna()
    df = df[test_date_2:end_date]

    buy_and_hold_agent = Buy_and_hold_Agent()
    sell_and_hold_agent = Sell_and_hold_Agent()

    # Run backtesting for both agents
    bah_results, _, benchmark_BAH = BF.run_backtesting(
        buy_and_hold_agent, 'BAH', [df], ['final_test'],
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, 10000, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    sah_results, _, benchmark_SAH = BF.run_backtesting(
        sell_and_hold_agent, 'SAH', [df], ['final_test'],
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, 10000, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    bah_results_prepared = prepare_backtest_results(bah_results, 'BAH')
    sah_results_prepared = prepare_backtest_results(sah_results, 'SAH')

    sah_results_prepared = sah_results_prepared.drop(('', 'Agent Generation'),
                                                     axis=1)  # drop the agent generation column
    bah_results_prepared = bah_results_prepared.drop(('', 'Agent Generation'),
                                                     axis=1)  # drop the agent generation column

    # Generate statistics for the final test results
    statistic_report = BF.generate_result_statistics(final_test_results, f'{agent.get_name()}_Action',
                                                     f'{agent.get_name()}_Balances', provision_sum_test_final,
                                                     look_back=look_back)

    statistic_report.update({'sell and hold final balance': sah_results_prepared[('SAH', 'Final Balance')][0],
                             'buy and hold final balance': bah_results_prepared[('BAH', 'Final Balance')][0],
                             'sell and hold sharpe ratio': sah_results_prepared[('SAH', 'Sharpe Ratio')][0],
                             'buy and hold sharpe ratio': bah_results_prepared[('BAH', 'Sharpe Ratio')][0]})

    print(f"Final Balance: {final_balance:.2f}")
    statistic_report = pd.DataFrame([statistic_report])
    #statistic_report.to_csv(fr'C:\Users\jmask\OneDrive\Pulpit\RL_magisterka\{tradable_markets}\statistic_report_{agent.get_name()}.csv', index=False)

    # Save final_test_results to a CSV file
    #final_test_results.to_csv(fr'C:\Users\jmask\OneDrive\Pulpit\RL_magisterka\{tradable_markets}\final_test_results_{agent.get_name()}.csv', index=False)

    # 12605 PogChamp benchmark add sharpe of this strategy # sharpe 1.55
    # 12100 good one
    print(statistic_report)
    print('end')