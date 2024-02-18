"""
PPO 7 added working backtesting


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

from data.function.load_data import load_data
from technical_analysys.add_indicators import add_indicators
from data.edit import normalize_data, standardize_data
import gym
from gym import spaces


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class BacktestShort:
    def __init__(self, df, look_back=20, variables=None, current_positions=True, tradable_markets='EURUSD',
                 provision=0.0001, agent=None, initial_balance=10000, environment=None):
        # Removed the call to the superclass constructor as it seems unnecessary
        self.df = df.reset_index(drop=True)
        self.df_original = df.copy()
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables
        self.current_positions = current_positions
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.agent = agent
        self.environment = environment

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        if self.current_positions:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.look_back + 1,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.look_back,), dtype=np.float32)

        self.reset()

    def backtest_short(self):
        self.df[('Capital_in', self.tradable_markets)] = 0.0
        self.df[('Capital', 'Strategy')] = 0.0
        self.df[('PnL', self.tradable_markets)] = 0.0
        self.df[('Provision', self.tradable_markets)] = 0.0
        self.df.loc[self.df.index[self.look_back - 1], ('Capital_in', self.tradable_markets)] = self.initial_balance
        self.df.loc[self.df.index[self.look_back - 1], ('Capital', 'Strategy')] = self.initial_balance
        self.df.loc[self.df.index[self.look_back - 1], ('PnL', self.tradable_markets)] = 0.0
        self.df.loc[self.df.index[self.look_back - 1], ('Provision', self.tradable_markets)] = 0.0
        self.df.loc[self.df.index[self.look_back - 1], ('Action', self.tradable_markets)] = 0

        for obs in range(len(self.df) - self.look_back):
            observation = self.environment.reset(obs)
            action = self.agent.choose_best_action(observation)
            self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] = action - 1
            self.df.loc[self.df.index[obs + self.look_back], ('PnL', self.tradable_markets)] = self.df.loc[
                                                                                                   self.df.index[
                                                                                                       obs + self.look_back - 1], (
                                                                                                   'Action',
                                                                                                   self.tradable_markets)] * \
                                                                                               self.df.loc[
                                                                                                   self.df.index[
                                                                                                       obs + self.look_back - 1], (
                                                                                                   'Capital_in',
                                                                                                   self.tradable_markets)] * (
                                                                                                           self.df.loc[
                                                                                                               self.df.index[
                                                                                                                   obs + self.look_back], (
                                                                                                               'Close',
                                                                                                               self.tradable_markets)] /
                                                                                                           self.df.loc[
                                                                                                               self.df.index[
                                                                                                                   obs + self.look_back - 1], (
                                                                                                               'Close',
                                                                                                               self.tradable_markets)] - 1) - \
                                                                                               self.df.loc[
                                                                                                   self.df.index[
                                                                                                       obs + self.look_back], (
                                                                                                   'Provision',
                                                                                                   self.tradable_markets)]
            self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')] = self.df.loc[self.df.index[
                obs + self.look_back - 1], ('Capital', 'Strategy')] + self.df.loc[self.df.index[obs + self.look_back], (
            'PnL', self.tradable_markets)]

            if self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] == self.df.loc[
                self.df.index[obs + self.look_back - 1], ('Action', self.tradable_markets)]:
                self.df.loc[self.df.index[obs + self.look_back], ('Capital_in', self.tradable_markets)] = self.df.loc[
                    self.df.index[obs + self.look_back - 1], ('Capital_in', self.tradable_markets)]

            else:
                if self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] == 0:
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital_in', self.tradable_markets)] = 0
                    self.df.loc[self.df.index[obs + self.look_back], ('Provision', self.tradable_markets)] = \
                    self.df.loc[
                        self.df.index[obs + self.look_back - 1], ('Capital_in', self.tradable_markets)] * provision
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')] = self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Capital',
                                                                                                    'Strategy')] - \
                                                                                                self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Provision',
                                                                                                    self.tradable_markets)]

                else:
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital_in', self.tradable_markets)] = \
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')]
                    self.df.loc[self.df.index[obs + self.look_back], ('Provision', self.tradable_markets)] = \
                    self.df.loc[self.df.index[obs + self.look_back - 1], (
                    'Capital_in', self.tradable_markets)] * provision * abs(
                        self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] -
                        self.df.loc[self.df.index[obs + self.look_back - 1], ('Action', self.tradable_markets)])
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')] = self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Capital',
                                                                                                    'Strategy')] - \
                                                                                                self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Provision',
                                                                                                    self.tradable_markets)]
        return self.df

    def calculate_trade_outcomes(self, actions, pnl):
        trade_outcomes = []
        current_trade_pnl = 0
        previous_action = None

        for action, pnl_value in zip(actions, pnl):
            if action != previous_action and action != 0:
                if previous_action is not None:  # End of a trade
                    trade_outcomes.append(current_trade_pnl)
                    current_trade_pnl = 0
            current_trade_pnl += pnl_value
            previous_action = action

        # Add the last trade if it's still open
        if current_trade_pnl != 0:
            trade_outcomes.append(current_trade_pnl)

        return trade_outcomes

    def report_short(self):
        report = {}
        df_log = self.df
        trade_outcomes = self.calculate_trade_outcomes(df_log[('Action', self.tradable_markets)],
                                                       df_log[('PnL', self.tradable_markets)])
        profitable_trades = len([pnl for pnl in trade_outcomes if pnl > 0])
        losing_trades = len([pnl for pnl in trade_outcomes if pnl < 0])
        report['Win Rate (%)'] = (profitable_trades / (losing_trades + profitable_trades)) * 100 if (losing_trades + profitable_trades) != 0 else 0
        report['Total PnL'] = self.df[('PnL', self.tradable_markets)].sum()
        report['Sharpe Ratio'] = self.df[('PnL', self.tradable_markets)].mean() / self.df[('PnL', self.tradable_markets)].std() if self.df[('PnL', self.tradable_markets)].std() != 0 else 0
        report['Total Return'] = self.df[('Capital', 'Strategy')].iloc[-1] / self.initial_balance - 1
        return report

    def plot(self):
        pass

    def reset(self):
        self.df = self.df_original.copy()
        self.current_position = 0

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



class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, dropout_rate=0.25):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.softmax(self.fc5(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, dropout_rate=0.25):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)
        q = self.fc5(x)
        return q


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.0025, gae_lambda=0.9, policy_clip=0.2, batch_size=1024, n_epochs=20, mini_batch_size=128):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO repair cuda
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.mini_batch_size = mini_batch_size

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

                # Actor Network Loss
                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_probs = dist.log_prob(batch_actions)
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                weighted_probs = batch_advantages * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                weighted_clipped_probs = clipped_probs * batch_advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Critic Network Loss
                critic_value = self.critic(batch_states).squeeze()
                returns = batch_advantages + batch_values
                critic_loss = nn.functional.mse_loss(critic_value, returns)

                # Gradient Calculation and Optimization Step
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

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

        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        with torch.no_grad():
            probs = self.actor(state)

        best_action = torch.argmax(probs, dim=1).item()
        return best_action


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, current_positions=True, tradble_markets='EURUSD', provision=0.0001):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = 10000
        self.current_position = 0
        self.variables = variables
        self.current_positions = current_positions
        self.tradable_markets = tradble_markets
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

    def reset(self, day=None):
        if day is not None:
            self.current_step = day + self.look_back
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
            if abs(mapped_action - self.current_position) == 2:
                provision = math.log(1 - 2 * self.provision)
            else:
                provision = math.log(1 - self.provision)
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
df = load_data(['EURUSD', 'USDJPY', 'EURJPY'], '1D')

indicators = [
    {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
    {"indicator": "RSI", "mkf": "EURJPY", "length": 14},
    {"indicator": "RSI", "mkf": "USDJPY", "length": 14},
    {"indicator": "ATR", "mkf": "EURUSD", "length": 24},
    {"indicator": "ATR", "mkf": "EURJPY", "length": 24},
    {"indicator": "SMA", "mkf": "EURUSD", "length": 100},
    {"indicator": "ATR", "mkf": "USDJPY", "length": 24},
    {"indicator": "MACD", "mkf": "EURUSD"},
    {"indicator": "Stochastic", "mkf": "USDJPY"},

]
add_indicators(df, indicators)
df = df.dropna()
start_date = '2017-01-01'
validation_date = '2021-01-01'
test_date = '2022-01-01'
df_train = df[start_date:validation_date]
df_validation = df[validation_date:test_date]
df_test = df[test_date:]

variables = [
    {"variable": ("RSI_14", "EURUSD"), "edit": "normalize"},
    {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "USDJPY"), "edit": "normalize"},
    {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    {"variable": ("Close", "EURJPY"), "edit": "normalize"}
]
tradable_markets = 'EURUSD'

print('number of train samples: ', len(df_train))
print('number of validation samples: ', len(df_validation))
print('number of test samples: ', len(df_test))

look_back = 20
provision = 0.001  # 0.001, cant be too high as it would not learn to trade
batch_size = 1024
epochs = 30  # 40
mini_batch_size = 256
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, current_positions=True, tradble_markets=tradable_markets, provision=provision)
agent = PPO_Agent(n_actions=env.action_space.n, input_dims=env.calculate_input_dims(), batch_size=batch_size, n_epochs=epochs, mini_batch_size=mini_batch_size)

num_episodes = 100  # 100

total_rewards = []
episode_durations = []
total_balances = []
index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
columns = ['Win Rate (%)', 'Total PnL', 'Sharpe Ratio', 'Total Return']
backtest_results = pd.DataFrame(index=index, columns=columns)

for episode in tqdm(range(num_episodes)):
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
        if len(agent.memory.states) == agent.memory.batch_size:
            agent.learn()

    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(cumulative_reward)
    episode_durations.append(episode_time)
    total_balances.append(env.balance)
    calculated_final_balance = initial_balance * math.exp(cumulative_reward)

    if episode % 1 == 0:
        print(f'\nEpisode: {episode + 1}')
        print(f"Episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
        print('----\n')
        # Backtesting on validation dataset
        validation_backtester = BacktestShort(df=df_validation, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets, provision=provision, agent=agent, initial_balance=10000,environment=env)
        validation_backtester.backtest_short()
        validation_report = validation_backtester.report_short()

        # Backtesting on test dataset
        test_backtester = BacktestShort(df=df_test, look_back=look_back, variables=variables, current_positions=True, tradable_markets=tradable_markets, provision=provision, agent=agent, initial_balance=10000, environment=env)
        test_backtester.backtest_short()
        test_report = test_backtester.report_short()

        # Store backtesting results
        backtest_results.loc[(episode, 'validation')] = validation_report
        backtest_results.loc[(episode, 'test')] = test_report

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

import matplotlib.pyplot as plt
'Win Rate (%)', 'Total PnL', 'Sharpe Ratio', 'Total Return'
# Extracting data for plotting
validation_pnl = backtest_results.loc[(slice(None), 'validation'), 'Total PnL']
test_pnl = backtest_results.loc[(slice(None), 'test'), 'Total PnL']

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), validation_pnl.values, label='Validation Total PnL', marker='o')
plt.plot(range(num_episodes), test_pnl.values, label='Test Total PnL', marker='x')

plt.title('Total PnL Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total PnL')
plt.legend()
plt.show()

validation_Win = backtest_results.loc[(slice(None), 'validation'), 'Win Rate (%)']
test_Win = backtest_results.loc[(slice(None), 'test'), 'Win Rate (%)']
# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), validation_Win.values, label='Validation Win Rate (%)', marker='o')
plt.plot(range(num_episodes), test_Win.values, label='Test Win Rate (%)', marker='x')

plt.title('Total Win Rate (%) Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Win Rate (%)')
plt.legend()
plt.show()

validation_Sharpe = backtest_results.loc[(slice(None), 'validation'), 'Sharpe Ratio']
test_Sharpe = backtest_results.loc[(slice(None), 'test'), 'Sharpe Ratio']
# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), validation_Sharpe.values, label='Validation Sharpe Ratio', marker='o')
plt.plot(range(num_episodes), test_Sharpe.values, label='Test Sharpe Ratio', marker='x')

plt.title('Sharpe Ratio Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.show()


validation_Return = backtest_results.loc[(slice(None), 'validation'), 'Total Return']
test_Return = backtest_results.loc[(slice(None), 'test'), 'Total Return']
# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), validation_Return.values, label='Validation Total Return', marker='o')
plt.plot(range(num_episodes), test_Return.values, label='Test Total Return', marker='x')

plt.title('Total Return Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.legend()
plt.show()

# final prediction agent
# TEST
df_test_probs = df_test.copy()
predictions_df = pd.DataFrame(index=df_test.index, columns=['Predicted_Action'])
test_env = Trading_Environment_Basic(df_test, look_back=look_back, variables=variables, current_positions=True, tradble_markets=tradable_markets)

for test_day in range(len(df_test) - test_env.look_back):
    observation = test_env.reset(test_day)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[test_day + test_env.look_back] = action

# Merge with df_test
df_test_with_predictions = df_test.copy()
df_test_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1


# final prediction with probabilities
test_env_probs = Trading_Environment_Basic(df_test_probs, look_back=look_back, variables=variables, current_positions=True, tradble_markets=tradable_markets)
action_probabilities = []

for test_day in range(len(df_test_probs) - test_env_probs.look_back):
    observation = test_env_probs.reset(test_day)  # Reset environment to the specific day
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
train_env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, current_positions=True, tradble_markets=tradable_markets)

for train_day in range(len(df_train) - train_env.look_back):
    observation = train_env.reset(train_day)
    action = agent.choose_best_action(observation)
    predictions_df.iloc[train_day + train_env.look_back] = action

# Merge with df_train
df_train_with_predictions = df_train.copy()
df_train_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1


train_env_probs = Trading_Environment_Basic(df_train_probs, look_back=look_back, variables=variables, current_positions=True, tradble_markets=tradable_markets)
action_probabilities = []

for train_day in range(len(df_train_probs) - train_env_probs.look_back):
    observation = train_env_probs.reset(train_day)  # Reset environment to the specific day
    probs = agent.get_action_probabilities(observation)
    action_probabilities.append(probs[0])

# Convert the list of probabilities to a DataFrame
probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Do_nothing', 'Long'])

# Join with the original train DataFrame
df_train_with_probabilities = df_train_probs.iloc[train_env_probs.look_back:].reset_index(drop=True)
df_train_with_probabilities = pd.concat([df_train_with_probabilities, probabilities_df], axis=1)

