"""
PPO 2.1

# TODO LIST
- Multiple Actors (Parallelization): Implement multiple actors that collect data in parallel. This can significantly speed up data collection and can lead to more diverse experience, helping in stabilizing training.
- Hyperparameter Tuning: Use techniques like grid search, random search, or Bayesian optimization to find the best set of hyperparameters.
- Noise Injection for Exploration: Inject noise into the policy or action space to encourage exploration. This can be particularly effective in continuous action spaces.
- Automated Architecture Search: Use techniques like neural architecture search (NAS) to automatically find the most suitable network architecture.
- try TFT transformer (Temporal Fusion Transformers transformer time series)

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

from data.function.load_data import load_data
from data.function.rolling_window import rolling_window_datasets
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns
from data.edit import normalize_data, standardize_data
import backtest.backtest_functions.functions as BF

# TODO add proper backtest function
def generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision=0.0001, initial_balance=10000, leverage=1):
    agent.actor.eval()
    agent.critic.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        # Create a validation environment
        validation_env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                                   tradable_markets=tradable_markets, provision=provision,
                                                   initial_balance=initial_balance, leverage=leverage)

        # Generate Predictions
        predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
        for validation_observation in range(len(df) - validation_env.look_back):
            observation = validation_env.reset(validation_observation)
            action, _, _ = agent.choose_action(observation)  # choose_action now returns continuous action
            predictions_df.iloc[validation_observation + validation_env.look_back] = action

        # Merge with original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action']

        # Backtesting
        balance = initial_balance
        current_position = 0.0  # Initial position is neutral (0)
        total_reward = 0  # Initialize total reward
        sum_of_position_changes = 0  # Initialize sum of position changes

        for i in range(look_back, len(df_with_predictions)):
            action = df_with_predictions['Predicted_Action'].iloc[i]
            current_price = df_with_predictions[('Close', mkf)].iloc[i - 1]
            next_price = df_with_predictions[('Close', mkf)].iloc[i]

            # Calculate log return
            log_return = math.log(next_price / current_price) if current_price != 0 else 0
            reward = action * log_return  # Reward is proportional to action

            # Apply leverage
            reward *= leverage

            # Calculate cost based on action
            provision_cost = abs(action - current_position) * math.log(1 - provision)

            reward += provision_cost

            # Update the balance
            balance *= math.exp(reward)

            # Calculate and accumulate the position change
            position_change = abs(action - current_position)
            sum_of_position_changes += position_change

            # Update current position
            current_position = action

            reward += provision * 99  # Add provision to reward

            total_reward = 100 * reward  # Scale reward for better learning

    # Ensure the agent's networks are back in training mode after evaluation
    agent.actor.train()
    agent.critic.train()

    return balance, total_reward, sum_of_position_changes, df_with_predictions

def backtest_wrapper(df, agent, mkf, look_back, variables, provision, initial_balance, leverage):
    return generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision, initial_balance, leverage)

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


# TODO rework network into TFT transformer network
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, dropout_rate=0.25):
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
        self.fc6 = nn.Linear(128, 1)
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
        x = torch.tanh(x)
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
    def __init__(self, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5):
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
        self.actor = ActorNetwork(input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self):
        self.actor.train()
        self.critic.train()

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
        """
        Selects an action based on the current policy. For continuous actions,
        the actor network outputs the mean of a Gaussian distribution from which we sample.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        # Actor network outputs mean of the action distribution
        mean = self.actor(state)
        std_dev = torch.tensor([0.5]).to(self.device)

        # Create a normal distribution and sample an action
        dist = torch.distributions.Normal(mean, std_dev)
        action = dist.sample()

        # Calculate the log probability of the action
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # Get the value from the critic network for the current state
        value = self.critic(state)
        action = torch.clamp(action, min=-0.99, max=0.99)

        return action.item(), log_prob.item(), value.item()


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
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32),
                                       high=np.array([1], dtype=np.float32),
                                       dtype=np.float32)
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
            else:  # Default none
                scaled_data = data

            scaled_observation.extend(scaled_data)

        if self.current_positions:
            scaled_observation = np.append(scaled_observation, (self.current_position+1)/2)

        return np.array(scaled_observation)

    def step(self, action):
        # Get current and next price
        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step + 1]

        # Calculate log return based on action
        log_return = math.log(next_price / current_price) if current_price != 0 else 0

        reward = log_return * action

        # Apply leverage to the base reward
        reward = reward * self.leverage

        provision = abs(action - self.current_position) * math.log(1 - self.provision) if action != 0 else 0

        reward += provision

        # Update the balance
        self.balance *= math.exp(reward)  # Update balance using exponential of reward before applying other penalties

        # Update current position and step
        self.current_position = action
        self.current_step += 1

        reward += provision * 99  # Add provision to reward

        final_reward = 100 * reward  # Scale reward for better learning

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

df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")]/100
df = df.dropna()
start_date = '2019-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[start_date:validation_date]
df_validation = df[validation_date:test_date]
df_test = df[test_date:]
variables = [
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "EURJPY"), "edit": "normalize"},
    {"variable": ("RSI_14", "EURUSD"), "edit": "None"},
    {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
]
tradable_markets = 'EURUSD'
window_size = '5Y'
starting_balance = 10000
look_back = 10
provision = 0.001  # 0.001, cant be too high as it would not learn to trade

# Training parameters
batch_size = 2048
epochs = 10  # 40
mini_batch_size = 64
leverage = 1
weight_decay = 0.0005
l1_lambda = 1e-5
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
agent = PPO_Agent(input_dims=env.calculate_input_dims(),
                  gamma=0.8,
                  alpha=0.05,  # learning rate for actor network
                  gae_lambda=0.8,  # lambda for generalized advantage estimation
                  policy_clip=0.25,  # clip parameter for PPO
                  entropy_coefficient=0.25,  # higher entropy coefficient encourages exploration
                  batch_size=batch_size,
                  n_epochs=epochs,
                  mini_batch_size=mini_batch_size,
                  weight_decay=weight_decay,
                  l1_lambda=l1_lambda)

num_episodes = 1000  # 100

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
    env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

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

    # Backtesting in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        validation_future = executor.submit(backtest_wrapper, df_validation, agent, 'EURUSD', look_back, variables,
                                            provision, starting_balance, leverage)
        test_future = executor.submit(backtest_wrapper, df_test, agent, 'EURUSD', look_back, variables, provision,
                                      starting_balance, leverage)

        # Retrieve results
        validation_balance, validation_total_rewards, validation_number_of_trades, validation_predictions = validation_future.result()
        test_balance, test_total_rewards, test_number_of_trades, test_predictions = test_future.result()

    backtest_results.loc[(episode, 'validation'), 'Final Balance'] = validation_balance
    backtest_results.loc[(episode, 'test'), 'Final Balance'] = test_balance
    backtest_results.loc[(episode, 'validation'), 'Final Reward'] = validation_total_rewards
    backtest_results.loc[(episode, 'test'), 'Final Reward'] = test_total_rewards
    backtest_results.loc[(episode, 'validation'), 'Number of Trades'] = validation_number_of_trades
    backtest_results.loc[(episode, 'test'), 'Number of Trades'] = test_number_of_trades
    backtest_results.loc[(episode, 'validation'), 'Dataset Index'] = dataset_index
    backtest_results.loc[(episode, 'test'), 'Dataset Index'] = dataset_index

    # results
    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(cumulative_reward)
    episode_durations.append(episode_time)
    total_balances.append(env.balance)

    print(f"\nCompleted learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
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

with ThreadPoolExecutor(max_workers=3) as executor:
    validation_future = executor.submit(backtest_wrapper, df_validation, agent, 'EURUSD', look_back, variables,
                                        provision, starting_balance, leverage)
    test_future = executor.submit(backtest_wrapper, df_test, agent, 'EURUSD', look_back, variables, provision,
                                  starting_balance, leverage)
    train_future = executor.submit(backtest_wrapper, df_train, agent, 'EURUSD', look_back, variables, provision,
                                   starting_balance, leverage)

    # Retrieve results
    validation_balance, validation_total_rewards, validation_number_of_trades, validation_predictions = validation_future.result()
    test_balance, test_total_rewards, test_number_of_trades, test_predictions = test_future.result()
    train_balance, train_total_rewards, train_number_of_trades, train_predictions = train_future.result()


# Assuming each DataFrame in the list has 'Predicted_Action' and 'Close' columns
dataframes = [validation_predictions, test_predictions, train_predictions]
validation_predictions=validation_predictions.dropna()
test_predictions=test_predictions.dropna()
train_predictions=train_predictions.dropna()

titles = ['Validation', 'Test', 'Train']  # Titles for the plots

for df, title in zip(dataframes, titles):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Predicted_Action
    axs[0].plot(df.index, df['Predicted_Action'], label='Predicted Action', color='blue')
    axs[0].set_title(f'Predicted Action Over Time ({title} Data)')
    axs[0].set_ylabel('Predicted Action')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Close Price
    axs[1].plot(df.index, df[('Close', tradable_markets)], label=f'Close Price', color='red')
    axs[1].set_title(f'Close Price Over Time ({title} Data)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Close Price')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

print('Max prediction validation:', max(validation_predictions['Predicted_Action']), 'Min prediction validation:', min(validation_predictions['Predicted_Action']))
print('Max prediction test:', max(test_predictions['Predicted_Action']), 'Min prediction test:', min(test_predictions['Predicted_Action']))
print('Max prediction train:', max(train_predictions['Predicted_Action']), 'Min prediction train:', min(train_predictions['Predicted_Action']))