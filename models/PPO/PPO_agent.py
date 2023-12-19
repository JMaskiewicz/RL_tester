"""
the newest version of PPO agent

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

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators
from data.edit import normalize_data, standardize_data

# TODO rework backtesting
def generate_predictions_and_backtest(df, agent, mkf, look_back, variables, current_positions, provision=0.0001, initial_balance=10000):
    # Create a validation environment
    validation_env = Trading_Environment_Basic(df, look_back, variables, current_positions, mkf)

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
        else:
            provision_cost = 0
        reward += provision_cost

        # Update the balance
        balance *= math.exp(reward)

        # Update current position
        current_position = action

    return balance


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


# TODO rework network into transformer network
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, dropout_rate=0.25):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, n_actions)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.fc1(state)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        if x.size(0) > 1:
            x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        if x.size(0) > 1:
            x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc6(x)
        x = self.softmax(x)
        return x

# TODO rework network into transformer network
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, dropout_rate=0.25):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, state):
        x = self.fc1(state)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        if x.size(0) > 1:
            x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        if x.size(0) > 1:
            x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)

        q = self.fc6(x)
        return q


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO repair cuda
        # self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.mini_batch_size = mini_batch_size
        self.entropy_coefficient = entropy_coefficient

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
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

                # Critic Network Loss
                critic_value = self.critic(batch_states).squeeze()
                returns = batch_advantages + batch_values
                critic_loss = nn.functional.mse_loss(critic_value, returns)

                # Gradient Calculation and Optimization Step
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            # Clear memory
            self.memory.clear_memory()  # TODO check if this is correct place or should be outside of the loop

    def choose_action(self, observation):
        """
        Selects an action based on the current policy and exploration noise.

        """
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
        """
        Returns the probabilities of each action for a given observation.
        """
        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        return probs.cpu().numpy()

    def choose_best_action(self, observation):
        """
        Selects the best action based on the current policy without exploration.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        with torch.no_grad():
            probs = self.actor(state)

        best_action = torch.argmax(probs, dim=1).item()
        return best_action


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, current_positions=True, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables
        self.current_positions = current_positions
        self.tradable_markets = tradable_markets
        self.provision = provision

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

    def render(self, mode='human'):
        if mode == 'human':
            window_start = max(0, self.current_step - self.look_back)
            window_end = self.current_step + 1
            plt.figure(figsize=(10, 6))

            # Plotting the price data
            plt.plot(self.df[('Close', self.tradable_markets)].iloc[window_start:window_end], label='Close Price')

            # Highlighting the current position: Buy (1), Sell (-1), Hold (0)
            if self.current_position == 1:
                plt.scatter(self.current_step, self.df[('Close', self.tradable_markets)].iloc[self.current_step],
                            color='green', label='Buy')
            elif self.current_position == -1:
                plt.scatter(self.current_step, self.df[('Close', self.tradable_markets)].iloc[self.current_step],
                            color='red', label='Sell')

            plt.title('Trading Environment State')
            plt.xlabel('Time Step')
            plt.ylabel('Price')
            plt.legend()
            plt.show()


# Example usage
df = load_data(['EURUSD', 'USDJPY', 'EURJPY'], '4H')

indicators = [
    {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
    {"indicator": "ATR", "mkf": "EURUSD", "length": 24},

]
add_indicators(df, indicators)
df = df.dropna()
start_date = '2016-01-01'
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

window_size = '6M'
look_back = 30
provision = 0.001  # 0.001, cant be too high as it would not learn to trade
batch_size = 2048
epochs = 20  # 40
mini_batch_size = 128
starting_balance = 10000
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance)
agent = PPO_Agent(n_actions=env.action_space.n,
                  input_dims=env.calculate_input_dims(),
                  gamma=0.8,
                  alpha=0.005,
                  gae_lambda=0.8,
                  policy_clip=0.2,
                  entropy_coefficient=0.01,
                  batch_size=batch_size,
                  n_epochs=epochs,
                  mini_batch_size=mini_batch_size)

num_episodes = 50  # 100

total_rewards = []
episode_durations = []
total_balances = []
index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
columns = ['Final Balance']
backtest_results = pd.DataFrame(index=index, columns=columns)

# Assuming df_train is your DataFrame
rolling_datasets = rolling_window_datasets(df_train, window_size=window_size,  look_back=look_back)

for episode in tqdm(range(num_episodes)):
    # Randomly select one dataset from the rolling datasets
    window_df = random.choice(rolling_datasets)

    print(f"\nEpisode {episode + 1}: Learning from dataset with Start Date = {window_df.index.min()}, End Date = {window_df.index.max()}, len = {len(window_df)}")
    # Create a new environment with the randomly selected window's data
    env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance)

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

    # Backtesting
    validation_balance = generate_predictions_and_backtest(df_validation, agent, 'EURUSD', look_back, variables, True, provision, initial_balance=starting_balance)
    test_balance = generate_predictions_and_backtest(df_test, agent, 'EURUSD', look_back, variables, True, provision, initial_balance=starting_balance)
    backtest_results.loc[(episode, 'validation'), 'Final Balance'] = validation_balance
    backtest_results.loc[(episode, 'test'), 'Final Balance'] = test_balance

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
validation_env = Trading_Environment_Basic(df_validation, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets)

for validation_observation in range(len(df_validation) - validation_env.look_back):
    observation = validation_env.reset(validation_observation)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[validation_observation + validation_env.look_back] = action

# Merge with df_validation
df_validation_with_predictions = df_validation.copy()
df_validation_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1

# final prediction with probabilities
validation_env_probs = Trading_Environment_Basic(df_validation_probs, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets)
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
test_env = Trading_Environment_Basic(df_test, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets)

for test_observation in range(len(df_test) - test_env.look_back):
    observation = test_env.reset(test_observation)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[test_observation + test_env.look_back] = action

# Merge with df_test
df_test_with_predictions = df_test.copy()
df_test_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1


# final prediction with probabilities
test_env_probs = Trading_Environment_Basic(df_test_probs, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets)
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
train_env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets)

for train_observation in range(len(df_train) - train_env.look_back):
    observation = train_env.reset(train_observation)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[train_observation + train_env.look_back] = action

# Merge with df_train
df_train_with_predictions = df_train.copy()
df_train_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1


train_env_probs = Trading_Environment_Basic(df_train_probs, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets)
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