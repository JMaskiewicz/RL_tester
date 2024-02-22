"""
DDQN 2.1

# TODO LIST
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- try transformer or TFT transformer (Temporal Fusion Transformers transformer time series)

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
# import backtest.backtest_functions.functions as BF

# TODO add proper backtest function
def make_predictions(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    for observation_idx in range(len(df) - env.look_back):
        observation = env.reset(observation_idx)
        action = agent.choose_best_action(observation)
        predictions_df.iloc[observation_idx + env.look_back] = action

    df_with_predictions = df.copy()
    df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1
    return df_with_predictions

def process_dataset_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    predictions = make_predictions(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    probabilities = calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    return predictions, probabilities

def calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    action_probabilities = []
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    for observation_idx in range(len(df) - env.look_back):
        observation = env.reset(observation_idx)
        probs = agent.get_action_probabilities(observation)
        assert probs.shape == (3,), f"Expected probs shape to be (3,), got {probs.shape}"
        action_probabilities.append(probs)

    probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Neutral', 'Long'])
    return probabilities_df

def calculate_probabilities_wrapper_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    return calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)

# TODO add proper backtest function
def generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision=0.0001, initial_balance=10000, leverage=1):
    # Switch to evaluation mode
    agent.q_policy.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                                   tradable_markets=mkf, provision=provision,
                                                   initial_balance=initial_balance, leverage=leverage)

        # Generate Predictions
        predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
        for observation_idx in range(len(df) - env.look_back):
            observation = env.reset(observation_idx)
            action = agent.choose_best_action(observation)
            predictions_df.iloc[observation_idx + env.look_back] = action

        # Merge with original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1

        # Backtesting
        balance = initial_balance
        current_position = 0  # Neutral position
        total_reward = 0  # Initialize total reward
        number_of_trades = 0

        for i in range(look_back, len(df_with_predictions)):
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

            # Apply leverage
            reward *= leverage

            # Calculate cost based on action and current position
            if action != current_position:
                if abs(action - current_position) == 2:
                    provision_cost = math.log(1 - 2 * provision)
                    number_of_trades += 2
                else:
                    provision_cost = math.log(1 - provision) if action != 0 else 0
                    number_of_trades += 1
            else:
                provision_cost = 0

            reward += provision_cost

            # Update the position
            current_position = action

            # Update the balance
            balance *= math.exp(reward)

            total_reward += reward * 1000  # Scale reward for better learning

    # Switch back to training mode
    agent.q_policy.train()

    return balance, total_reward, number_of_trades

def backtest_wrapper(df, agent, mkf, look_back, variables, provision, initial_balance, leverage):
    return generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision, initial_balance, leverage)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, dropout_rate=0.125):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO repair cuda
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

        print('Learning...')
        self.replace_target_network()

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

    def get_action_probabilities(self, observation):
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


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000, leverage=1):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage

        # Define action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.Discrete(3)

        # Define observation space based on the look_back period and the number of variables
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(look_back + 1,),  # +1 for current position
                                            dtype=np.float32)

        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        input_dims += 1  # Add one more dimension for current position
        return input_dims

    def reset(self, observation_idx=None):
        if observation_idx is not None:
            self.current_step = observation_idx + self.look_back
        else:
            self.current_step = self.look_back

        self.balance = self.initial_balance
        self.current_position = 0
        self.done = False
        return self._next_observation()

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
            else:  # Default none
                scaled_data = data

            scaled_observation.extend(scaled_data)

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
        reward *= self.leverage

        # Calculate cost based on action and current position
        if mapped_action != self.current_position:
            if abs(mapped_action - self.current_position) == 2:
                provision = math.log(1 - 2 * self.provision)
            else:
                provision = math.log(1 - self.provision)
        else:
            provision = 0

        reward += provision

        # Update the balance
        self.balance *= math.exp(reward)  # Update balance using exponential of reward before applying other penalties

        # Update current position and step
        self.current_position = mapped_action
        self.current_step += 1
        """
        multiple reward by X to make it more significant and to make it easier for the agent to learn, 
        without this the agent would not learn as the reward is too close to 0
        """

        final_reward = 1000 * reward

        # Check if the episode is done
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._next_observation(), final_reward, self.done, {}

# Example usage
# Stock market variables
df = load_data(['EURUSD', 'USDJPY', 'EURJPY'], '1D')

indicators = [
    {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
    {"indicator": "ATR", "mkf": "EURUSD", "length": 24},
    {"indicator": "MACD", "mkf": "EURUSD"},
    {"indicator": "Stochastic", "mkf": "EURUSD"},]

add_indicators(df, indicators)
df = df.dropna()
start_date = '2008-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[start_date:validation_date]
df_validation = df[validation_date:test_date]
df_test = df[test_date:]
variables = [
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "EURJPY"), "edit": "normalize"},
    {"variable": ("RSI_14", "EURUSD"), "edit": "normalize"},
    {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
]
tradable_markets = 'EURUSD'
window_size = '1Y'
starting_balance = 10000
look_back = 20
provision = 0.001  # 0.001, cant be too high as it would not learn to trade

# Training parameters
batch_size = 1024
epochs = 10  # 40
mini_batch_size = 128
leverage = 1
weight_decay = 0.0005
l1_lambda = 0.00005
num_episodes = 1000  # at 500 starts over fitting probably
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
agent = DDQN_Agent(input_dims=env.calculate_input_dims(),
                   n_actions=env.action_space.n,
                   epochs=epochs,
                   mini_batch_size=mini_batch_size,
                   policy_alpha=0.001,
                   target_alpha=0.0005,
                   gamma=0.9,
                   epsilon=1.0,
                   epsilon_dec=0.95,
                   epsilon_end=0,
                   mem_size=100000,
                   batch_size=batch_size,
                   replace=5,  # num_episodes // 4
                   weight_decay=weight_decay,
                   l1_lambda=l1_lambda)

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
        if agent.memory.mem_cntr >= agent.batch_size:
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

    # calculate probabilities
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_train = executor.submit(calculate_probabilities_wrapper_DQN, df_train, Trading_Environment_Basic, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
        future_validation = executor.submit(calculate_probabilities_wrapper_DQN, df_validation, Trading_Environment_Basic,agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
        future_test = executor.submit(calculate_probabilities_wrapper_DQN, df_test, Trading_Environment_Basic, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)

        train_probs = future_train.result()
        validation_probs = future_validation.result()
        test_probs = future_test.result()

    episode_probabilities['train'].append(train_probs[['Short', 'Neutral', 'Long']].to_dict(orient='list'))
    episode_probabilities['validation'].append(validation_probs[['Short', 'Neutral', 'Long']].to_dict(orient='list'))
    episode_probabilities['test'].append(test_probs[['Short', 'Neutral', 'Long']].to_dict(orient='list'))

    # results
    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(cumulative_reward)
    episode_durations.append(episode_time)
    total_balances.append(env.balance)

    print(f"\nCompleted learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds, Agent Epsilon: {agent.get_epsilon():.4f}")
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
# predictions and probabilities for train, validation and test calculated in parallel
with ThreadPoolExecutor() as executor:
    futures = []
    datasets = [df_train, df_validation, df_test]
    for df in datasets:
        futures.append(executor.submit(process_dataset_DQN, df, Trading_Environment_Basic, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage))

    results = [future.result() for future in futures]

# Unpack results
df_train_with_predictions, df_train_with_probabilities = results[0]
df_validation_with_predictions, df_validation_with_probabilities = results[1]
df_test_with_predictions, df_test_with_probabilities = results[2]

# plotting probabilities
plt.figure(figsize=(16, 6))
plt.plot(df_validation_with_probabilities['Short'], label='Short', color='red')
plt.plot(df_validation_with_probabilities['Neutral'], label='Neutral', color='blue')
plt.plot(df_validation_with_probabilities['Long'], label='Long', color='green')

plt.title('Action Probabilities Over Time for Validation Set')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(df_train_with_probabilities['Short'], label='Short', color='red')
plt.plot(df_train_with_probabilities['Neutral'], label='Neutral', color='blue')
plt.plot(df_train_with_probabilities['Long'], label='Long', color='green')

plt.title('Action Probabilities Over Time for Train Set')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(df_test_with_probabilities['Short'], label='Short', color='red')
plt.plot(df_test_with_probabilities['Neutral'], label='Neutral', color='blue')
plt.plot(df_test_with_probabilities['Long'], label='Long', color='green')

plt.title('Action Probabilities Over Time for Test Set')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.show()

###
import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import webbrowser
import pandas as pd


def get_ohlc_data(selected_dataset, market):
    dataset_mapping = {
        'train': df_train,
        'validation': df_validation,
        'test': df_test
    }
    data = dataset_mapping[selected_dataset]

    # Construct the column tuples based on the MultiIndex structure
    ohlc_columns = [('Open', market), ('High', market), ('Low', market), ('Close', market)]

    # Extract the OHLC data for the specified market
    ohlc_data = data.loc[:, ohlc_columns]

    # Reset index to turn 'Date' into a column
    ohlc_data = ohlc_data.reset_index()

    # Flatten the MultiIndex for columns, and map each OHLC column correctly
    ohlc_data.columns = ['Time'] + [col[0] for col in ohlc_data.columns[1:]]

    # Ensure 'Time' column is in datetime format
    ohlc_data['Time'] = pd.to_datetime(ohlc_data['Time'])

    return ohlc_data

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[
            {'label': 'Train', 'value': 'train'},
            {'label': 'Validation', 'value': 'validation'},
            {'label': 'Test', 'value': 'test'}
        ],
        value='train'
    ),
    dcc.Input(id='episode-input', type='number', value=0, min=0, step=1),
    dcc.Graph(id='probability-plot'),
    dcc.Graph(id='ohlc-plot')
])

# Callback for updating the probability plot
@app.callback(
    Output('probability-plot', 'figure'),
    [Input('dataset-dropdown', 'value'), Input('episode-input', 'value')]
)

def update_probability_plot(selected_dataset, selected_episode):
    data = episode_probabilities[selected_dataset][selected_episode]
    x_values = list(range(len(data['Short'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=data['Short'], mode='lines', name='Short'))
    fig.add_trace(go.Scatter(x=x_values, y=data['Neutral'], mode='lines', name='Neutral'))
    fig.add_trace(go.Scatter(x=x_values, y=data['Long'], mode='lines', name='Long'))
    fig.update_layout(title='Action Probabilities Over Episodes', xaxis_title='Time', yaxis_title='Probability', yaxis=dict(range=[0, 1]))
    return fig

@app.callback(
    Output('ohlc-plot', 'figure'),
    [Input('dataset-dropdown', 'value')]
)


def update_ohlc_plot(selected_dataset, market='EURUSD'):
    ohlc_data = get_ohlc_data(selected_dataset, market)

    if ohlc_data.empty:
        return go.Figure(layout=go.Layout(title="No OHLC data available."))

    # Create OHLC plot
    ohlc_fig = go.Figure(data=[go.Ohlc(
        x=ohlc_data['Time'],
        open=ohlc_data['Open'],
        high=ohlc_data['High'],
        low=ohlc_data['Low'],
        close=ohlc_data['Close']
    )])

    ohlc_fig.update_layout(title='OHLC Data', xaxis_title='Time', yaxis_title='Price')

    return ohlc_fig

app.run_server(debug=True, port=8062)

# Open the web browser
webbrowser.open("http://127.0.0.1:8062/")

# Prepare the example observation
observation_window = df_train.iloc[60:60+look_back]
processed_observation = []

for variable in variables:
    data = observation_window[variable['variable']].values
    if variable['edit'] == 'standardize':
        processed_data = standardize_data(data)
    elif variable['edit'] == 'normalize':
        processed_data = normalize_data(data)
    else:
        processed_data = data
    processed_observation.extend(processed_data)

# Convert to numpy array
processed_observation = np.array(processed_observation)

def get_probabilities_for_position(current_position):
    observation_with_position = np.append(processed_observation, (current_position+1)/2)
    observation_with_position = observation_with_position.reshape(1, -1)
    return agent.get_action_probabilities(observation_with_position)

# Get probabilities for each position
probabilities_short = get_probabilities_for_position(-1)
probabilities_neutral = get_probabilities_for_position(0)
probabilities_long = get_probabilities_for_position(1)

# Print or return the results
print("Probabilities for Short Position:", probabilities_short)
print("Probabilities for Neutral Position:", probabilities_neutral)
print("Probabilities for Long Position:", probabilities_long)

print('end')
