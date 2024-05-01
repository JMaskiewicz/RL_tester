"""
PPO 3.3

# TODO LIST
make update to align with the new functions
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

from data.function.load_data import load_data_parallel
from data.function.rolling_window import rolling_window_datasets
from data.function.edit import normalize_data, standardize_data
from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
import backtest.backtest_functions.functions as BF
from functions.utilis import save_actor_critic_model


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
        self.batch_size = batch_size
        self.clear_memory()

        self.device = device

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        indices = indices[torch.randperm(n_states)]  # Shuffle indices
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        self.actions.append(torch.tensor(action, dtype=torch.long).unsqueeze(0))
        self.probs.append(torch.tensor(probs, dtype=torch.float).unsqueeze(0))
        self.vals.append(torch.tensor(vals, dtype=torch.float).unsqueeze(0))
        self.rewards.append(torch.tensor(reward, dtype=torch.float).unsqueeze(0))
        self.dones.append(torch.tensor(done, dtype=torch.bool).unsqueeze(0))

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def stack_tensors(self):
        self.states = torch.cat(self.states, dim=0).to(self.device)
        self.actions = torch.cat(self.actions, dim=0).to(self.device)
        self.probs = torch.cat(self.probs, dim=0).to(self.device)
        self.vals = torch.cat(self.vals, dim=0).to(self.device)
        self.rewards = torch.cat(self.rewards, dim=0).to(self.device)
        self.dones = torch.cat(self.dones, dim=0).to(self.device)

# TODO rework network into TFT transformer network
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, dropout_rate=1/16):
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
    def __init__(self, input_dims, dropout_rate=1/16):
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
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.actor = ActorNetwork(n_actions, input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.memory = PPOMemory(batch_size, self.device)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self):
        print('Learning... CHECK')
        self.actor.train()
        self.critic.train()

        self.memory.stack_tensors()

        for _ in range(self.n_epochs):
            # Generating the data for the entire batch
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, _ = self.memory.generate_batches()

            # Convert arrays to tensors
            state_arr = state_arr.clone().detach().to(self.device)
            action_arr = action_arr.clone().detach().to(self.device)
            old_prob_arr = old_prob_arr.clone().detach().to(self.device)
            vals_arr = vals_arr.clone().detach().to(self.device)
            reward_arr = reward_arr.clone().detach().to(self.device)
            dones_arr = dones_arr.clone().detach().to(self.device)

            # Convert arrays to tensors
            if isinstance(vals_arr, torch.Tensor):
                values = vals_arr.clone().detach().to(self.device)
            else:
                values = torch.tensor(vals_arr, dtype=torch.float).to(self.device)

            # Compute advantages and discounted rewards using the new vectorized method
            advantages, discounted_rewards = self.compute_discounted_rewards(reward_arr, values.cpu().numpy(), dones_arr)
            advantages = advantages.clone().detach().to(self.device)
            discounted_rewards = discounted_rewards.clone().detach().to(self.device)

            # Creating mini-batches
            num_samples = len(state_arr)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.mini_batch_size):
                # Extract indices for the mini-batch
                minibatch_indices = indices[start_idx:start_idx + self.mini_batch_size]

                # Extract data for the current mini-batch
                batch_states = state_arr[minibatch_indices].clone().detach().to(self.device)
                batch_actions = action_arr[minibatch_indices].clone().detach().to(self.device)
                batch_old_probs = old_prob_arr[minibatch_indices].clone().detach().to(self.device)
                batch_advantages = advantages[minibatch_indices]
                batch_returns = discounted_rewards[minibatch_indices]

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
                critic_loss = nn.functional.mse_loss(critic_value, batch_returns)
                l1_loss_critic = sum(torch.sum(torch.abs(param)) for param in self.critic.parameters())
                critic_loss += self.l1_lambda * l1_loss_critic

                # Gradient Calculation and Optimization Step
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Clear memory
        self.memory.clear_memory()

    def compute_discounted_rewards(self, rewards, values, dones):
        n = len(rewards)
        discounted_rewards = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

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
        Selects the best action based on the current policy without exploration by using
        the probabilities from get_action_probabilities function.
        """
        # Get action probabilities for the given observation
        action_probs = self.get_action_probabilities(observation)

        # Use np.argmax to select the action with the highest probability
        best_action = np.argmax(action_probs)

        return best_action

    def get_name(self):
        """
        Returns the class name of the instance.
        """
        return self.__class__.__name__


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000, leverage=1, reward_scaling=1):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage
        self.reward_scaling = reward_scaling

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
        reward = log_return * mapped_action * self.leverage

        # Apply leverage to the base reward
        reward *= self.leverage

        # Calculate cost based on action and current position
        if mapped_action != self.current_position:
            if abs(mapped_action) == 1:
                provision = math.log(1 - self.provision)
            else:
                provision = 0
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

        final_reward = reward * self.reward_scaling

        # Check if the episode is done
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._next_observation(), final_reward, self.done, {}

# Example usage
# Stock market variables
if __name__ == '__main__':
    df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD'], '1D')

    indicators = [
        {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
        {"indicator": "ATR", "mkf": "EURUSD", "length": 24},
        {"indicator": "MACD", "mkf": "EURUSD"},
        {"indicator": "Stochastic", "mkf": "EURUSD"},]

    add_indicators(df, indicators)
    add_time_sine_cosine(df, '1W')
    add_time_sine_cosine(df, '1M')
    df[("sin_time_1W", "")] = df[("sin_time_1W", "")]/2 + 0.5
    df[("cos_time_1W", "")] = df[("cos_time_1W", "")]/2 + 0.5
    df[("sin_time_1M", "")] = df[("sin_time_1M", "")]/2 + 0.5
    df[("cos_time_1M", "")] = df[("cos_time_1M", "")]/2 + 0.5
    df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")]/100

    df = df.dropna()
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
        {"variable": ("RSI_14", "EURUSD"), "edit": "normalize"},
        {"variable": ("ATR_24", "EURUSD"), "edit": "normalize"},
        #{"variable": ("sin_time_1W", ""), "edit": None},  # should be effecting neutral position?
        #{"variable": ("cos_time_1W", ""), "edit": None},
    ]

    tradable_markets = 'EURUSD'
    window_size = '1Y'
    starting_balance = 10000
    look_back = 20
    # Provision is the cost of trading, it is a percentage of the trade size, current real provision on FOREX is 0.0001
    provision = 0.0001  # 0.001, cant be too high as it would not learn to trade

    # Training parameters
    batch_size = 1024
    epochs = 1  # 40
    mini_batch_size = 128
    leverage = 10
    weight_decay = 0.00001
    l1_lambda = 1e-7
    reward_scaling = 1000
    num_episodes = 200  # 100
    # Create the environment
    env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage, reward_scaling=reward_scaling)
    agent = PPO_Agent(n_actions=env.action_space.n,
                      input_dims=env.calculate_input_dims(),
                      gamma=0.9,
                      alpha=0.0005,  # learning rate for actor network
                      gae_lambda=0.8,  # lambda for generalized advantage estimation
                      policy_clip=0.2,  # clip parameter for PPO
                      entropy_coefficient=0.5,  # higher entropy coefficient encourages exploration
                      batch_size=batch_size,
                      n_epochs=epochs,
                      mini_batch_size=mini_batch_size,
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
        env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage, reward_scaling=reward_scaling)

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
            if len(agent.memory.states) >= agent.memory.batch_size:  # or done:
                agent.learn()
                agent.memory.clear_memory()

        # Backtesting in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            validation_future = executor.submit(BF.backtest_wrapper_AC, df_validation, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage, reward_scaling, Trading_Environment_Basic=Trading_Environment_Basic)
            test_future = executor.submit(BF.backtest_wrapper_AC, df_test, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage, reward_scaling, Trading_Environment_Basic=Trading_Environment_Basic)

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
            future_train = executor.submit(BF.calculate_probabilities_wrapper_AC, df_train, Trading_Environment_Basic, agent,look_back, variables, tradable_markets, provision, starting_balance, leverage)
            future_validation = executor.submit(BF.calculate_probabilities_wrapper_AC, df_validation, Trading_Environment_Basic,agent, look_back, variables, tradable_markets, provision, starting_balance,leverage)
            future_test = executor.submit(BF.calculate_probabilities_wrapper_AC, df_test, Trading_Environment_Basic, agent,look_back, variables, tradable_markets, provision, starting_balance, leverage)

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

        print(f"\nCompleted learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
        print("-----------------------------------")

    # Save the trained agent
    # save_actor_critic_model(agent.actor, agent.critic, 'PPO_NN_agent_final') # TODO repair

    # Plotting the results after all episodes
    print(backtest_results)

    # final prediction agent
    # predictions and probabilities for train, validation and test calculated in parallel
    from backtest.plots.plot import plot_all
    from backtest.plots.OHLC_probability_plot import OHLC_probability_plot

    with ThreadPoolExecutor() as executor:
        futures = []
        datasets = [df_train, df_validation, df_test]
        for df in datasets:
            futures.append(executor.submit(BF.process_dataset_AC, df, Trading_Environment_Basic, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage))

        results = [future.result() for future in futures]

    # Unpack results
    df_train_with_predictions, df_train_with_probabilities = results[0]
    df_validation_with_predictions, df_validation_with_probabilities = results[1]
    df_test_with_predictions, df_test_with_probabilities = results[2]

    # Extracting data for plotting
    validation_pnl = backtest_results.loc[(slice(None), 'validation'), 'Final Balance']
    test_pnl = backtest_results.loc[(slice(None), 'test'), 'Final Balance']

    # plotting everything
    probabilities_sets = {
        'Validation': df_validation_with_probabilities,
        'Train': df_train_with_probabilities,
        'Test': df_test_with_probabilities
    }

    plot_all(
        total_rewards=total_rewards,
        episode_durations=episode_durations,
        total_balances=total_balances,
        num_episodes=num_episodes,
        validation_pnl=validation_pnl,
        test_pnl=test_pnl,
        probabilities_sets=probabilities_sets,
        plot_rewards=True,
        plot_durations=True,
        plot_balances=True,
        plot_pnl=True,
        plot_probabilities=True,
        model_name=agent.get_name(),
    )

    ###
    OHLC_probability_plot(df_train, df_validation, df_test, episode_probabilities, portnumber=8060)

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
