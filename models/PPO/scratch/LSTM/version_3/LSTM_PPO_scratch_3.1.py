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
import torch.nn.functional as F

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

            total_reward += 100 * reward  # Scale reward for better learning

    # Ensure the agent's networks are back in training mode after evaluation
    agent.actor.train()
    agent.critic.train()

    return balance, total_reward, number_of_trades

def backtest_wrapper(df, agent, mkf, look_back, variables, provision, initial_balance, leverage):
    return generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision, initial_balance, leverage)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class PPOMemory_LSTM:
    def __init__(self, batch_size, device):
        self.states = None
        self.probs = None
        self.actions = None
        self.vals = None
        self.rewards = None
        self.dones = None
        self.static_inputs = None  # Container for static inputs
        self.batch_size = batch_size
        self.clear_memory()

        self.device = device

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = torch.arange(0, n_states, self.batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        indices = indices[torch.randperm(n_states)]
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, self.static_inputs, batches

    def store_memory(self, state, action, probs, vals, reward, done, static_input):
        self.states.append(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        self.actions.append(torch.tensor(action, dtype=torch.long).unsqueeze(0))
        self.probs.append(torch.tensor(probs, dtype=torch.float).unsqueeze(0))
        self.vals.append(torch.tensor(vals, dtype=torch.float).unsqueeze(0))
        self.rewards.append(torch.tensor(reward, dtype=torch.float).unsqueeze(0))
        self.dones.append(torch.tensor(done, dtype=torch.bool).unsqueeze(0))
        self.static_inputs.append(torch.tensor(static_input, dtype=torch.float).unsqueeze(0))

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.static_inputs = []

    def stack_tensors(self):
        self.states = torch.cat(self.states, dim=0).to(self.device)
        self.actions = torch.cat(self.actions, dim=0).to(self.device)
        self.probs = torch.cat(self.probs, dim=0).to(self.device)
        self.vals = torch.cat(self.vals, dim=0).to(self.device)
        self.rewards = torch.cat(self.rewards, dim=0).to(self.device)
        self.dones = torch.cat(self.dones, dim=0).to(self.device)
        self.static_inputs = torch.cat(self.static_inputs, dim=0).to(self.device)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(True),
            nn.Dropout(1/16),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(1/16),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(1 / 16),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(1/16),
            nn.Linear(256, 1)
        )

    def forward(self, lstm_output):
        energy = self.projection(lstm_output)
        weights = F.softmax(energy, dim=1)
        outputs = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class LSTM_NetworkBase(nn.Module):
    def __init__(self, input_dims, static_dim, hidden_size=2048, n_layers=2):
        super(LSTM_NetworkBase, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size,
                            batch_first=True, num_layers=n_layers, dropout=0.2)
        self.self_attention = SelfAttention(hidden_size + static_dim)

    def forward(self, state, static_input):
        lstm_output, _ = self.lstm(state)
        if lstm_output.dim() == 2:
            lstm_output = lstm_output.unsqueeze(0)

        # Process static input
        batch_size, seq_len, _ = lstm_output.shape
        static_input = static_input.unsqueeze(-1)
        static_input_expanded = static_input.expand(batch_size, seq_len, -1)

        # Combine LSTM output and static input
        combined_input = torch.cat((lstm_output, static_input_expanded), dim=2)
        attention_output, _ = self.self_attention(combined_input)

        return attention_output


class LSTM_ActorNetwork(LSTM_NetworkBase):
    def __init__(self, n_actions, input_dims, static_dim, hidden_size=2048, dropout=1/16):
        super(LSTM_ActorNetwork, self).__init__(input_dims, static_dim, hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.policy = nn.Linear(256, n_actions)

    def forward(self, state, static_input):
        attention_output = super().forward(state, static_input).squeeze(0)
        x = self.fc1(attention_output)
        action_probs = torch.softmax(self.policy(x), dim=-1)
        return action_probs


class LSTM_CriticNetwork(LSTM_NetworkBase):
    def __init__(self, input_dims, static_dim, hidden_size=2048, dropout=1/16):
        super(LSTM_CriticNetwork, self).__init__(input_dims, static_dim, hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.value = nn.Linear(256, 1)

    def forward(self, state, static_input):
        attention_output = super().forward(state, static_input).squeeze(0)
        x = self.fc1(attention_output)
        value = self.value(x)
        return value


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.2, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5):
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
        self.static_dim = 1

        # Initialize the actor and critic networks
        self.actor = LSTM_ActorNetwork(n_actions, input_dims, self.static_dim).to(self.device)
        self.critic = LSTM_CriticNetwork(input_dims, self.static_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha, weight_decay=weight_decay)

        self.memory = PPOMemory_LSTM(batch_size, self.device)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self):
        print('Learning... CHECK')
        self.actor.train()
        self.critic.train()

        self.memory.stack_tensors()  # Ensure this function is compatible with sequences

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, static_inputs_arr = self.memory.generate_batches()

            # Ensure all tensors are on the correct device
            state_arr = state_arr.to(self.device)
            action_arr = action_arr.to(self.device)
            old_prob_arr = old_prob_arr.to(self.device)
            vals_arr = vals_arr.to(self.device)
            reward_arr = reward_arr.to(self.device)
            dones_arr = dones_arr.to(self.device)
            static_inputs_arr = static_inputs_arr.to(self.device)

            values = vals_arr.clone().detach()

            advantages, discounted_rewards = self.compute_discounted_rewards(reward_arr, values.cpu().numpy(),
                                                                             dones_arr)
            advantages = advantages.to(self.device)
            discounted_rewards = discounted_rewards.to(self.device)

            for start_idx in range(0, len(state_arr), self.mini_batch_size):
                minibatch_indices = torch.arange(start_idx, min(start_idx + self.mini_batch_size, len(state_arr)))
                batch_states = state_arr[minibatch_indices]
                batch_actions = action_arr[minibatch_indices]
                batch_old_probs = old_prob_arr[minibatch_indices]
                batch_advantages = advantages[minibatch_indices]
                batch_returns = discounted_rewards[minibatch_indices]
                batch_static_inputs = static_inputs_arr[minibatch_indices]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Pass both dynamic and static inputs to the networks
                probs = self.actor(batch_states, batch_static_inputs)
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
                critic_value = self.critic(batch_states, batch_static_inputs).squeeze()
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

    def choose_action(self, observation, static_input):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        static_input = torch.tensor([static_input], dtype=torch.float).to(self.device)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        probs = self.actor(state, static_input)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state, static_input)

        return action.item(), log_prob.item(), value.item()

    def get_action_probabilities(self, observation, static_input):
        """
        Returns the probabilities of each action for a given observation.
        Observation should be formatted as a sequence for the LSTM.
        Static input should be provided if the network expects additional context.
        """
        # Convert the observation to a PyTorch tensor and add required dimensions (batch and sequence)
        # Assuming observation is a sequence of observations suitable for LSTM
        observation = torch.tensor([observation], dtype=torch.float).to(
            self.device)  # Shape: (1, sequence_length, feature_size)

        # Prepare static input, ensure it's a tensor and add batch dimension
        static_input = torch.tensor([static_input], dtype=torch.float).to(
            self.device)  # Shape: (1, static_feature_size)

        with torch.no_grad():
            # Pass both the sequence of observations and static input to the actor network
            probs = self.actor(observation, static_input)

        # Move the probabilities back to CPU and convert to numpy for external processing
        return probs.cpu().numpy()

    def choose_best_action(self, observation, static_input):
        """
        Selects the best action based on the current policy without exploration.
        """
        # Ensure observation is in the correct shape and type
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)  # Add batch dimension
        static_input = torch.tensor([static_input], dtype=torch.float).to(
            self.device)  # Ensure static input is tensor and on the correct device

        with torch.no_grad():
            probs = self.actor(observation, static_input)  # Pass both dynamic and static inputs
        best_action = torch.argmax(probs, dim=-1).item()  # Get the action with the highest probability

        return best_action


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

        # Collect a sequence of observations
        observation_sequence = []
        for variable in self.variables:
            data = self.df[variable['variable']].iloc[start:end].values
            if variable['edit'] == 'standardize':
                scaled_data = standardize_data(data)
            elif variable['edit'] == 'normalize':
                scaled_data = normalize_data(data)
            else:  # Default none
                scaled_data = data
            observation_sequence.append(scaled_data)

        # Stack along the new axis to create a sequence
        observation_sequence = np.stack(observation_sequence, axis=1)

        # Add additional features if required (e.g., current position)
        additional_features = np.full((observation_sequence.shape[0], 1), (self.current_position + 1) / 2)

        # Combine the observation sequence with the additional features
        combined_observation = np.hstack((observation_sequence, additional_features))

        return combined_observation

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
batch_size = 512
epochs = 1  # 40
mini_batch_size = 128
leverage = 1
weight_decay = 0.00001
l1_lambda = 1e-7
# Create the environment
env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)
agent = PPO_Agent(n_actions=env.action_space.n,
                  input_dims=env.calculate_input_dims(),
                  gamma=0.9,
                  alpha=0.005,  # learning rate for actor network
                  gae_lambda=0.8,  # lambda for generalized advantage estimation
                  policy_clip=0.25,  # clip parameter for PPO
                  entropy_coefficient=0.5,  # higher entropy coefficient encourages exploration
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

    print(
        f"\nEpisode {episode + 1}: Learning from dataset with Start Date = {window_df.index.min()}, End Date = {window_df.index.max()}, len = {len(window_df)}")

    # Create a new environment with the selected window's data
    env = Trading_Environment_Basic(window_df, look_back=look_back, variables=variables,
                                    tradable_markets=tradable_markets, provision=provision,
                                    initial_balance=starting_balance, leverage=leverage)

    observation, static_input = env.reset()  # Ensure reset returns both observation and static input if required
    done = False
    cumulative_reward = 0
    start_time = time.time()

    while not done:
        action, prob, val = agent.choose_action(observation,
                                                static_input)  # Pass both dynamic observation and static input
        observation_, reward, done, info = env.step(action)

        # Store transition with static input if required
        agent.store_transition(observation, action, prob, val, reward, done,
                               static_input)  # Add static input to stored memory if needed

        observation = observation_  # Update observation for the next step
        cumulative_reward += reward

        # Check if enough data is collected or if the episode ends
        if len(agent.memory.states) >= agent.memory.batch_size:
            agent.learn()
            agent.memory.clear_memory()

    # Parallel backtesting for validation and testing datasets
    with ThreadPoolExecutor(max_workers=2) as executor:
        validation_future = executor.submit(backtest_wrapper, df_validation, agent, 'EURUSD', look_back, variables,
                                            provision, starting_balance, leverage)
        test_future = executor.submit(backtest_wrapper, df_test, agent, 'EURUSD', look_back, variables, provision,
                                      starting_balance, leverage)

        validation_balance, validation_total_rewards, validation_number_of_trades = validation_future.result()
        test_balance, test_total_rewards, test_number_of_trades = test_future.result()

    # Update backtest results
    backtest_results.loc[(episode, 'validation'), 'Final Balance'] = validation_balance
    backtest_results.loc[(episode, 'test'), 'Final Balance'] = test_balance
    backtest_results.loc[(episode, 'validation'), 'Final Reward'] = validation_total_rewards
    backtest_results.loc[(episode, 'test'), 'Final Reward'] = test_total_rewards
    backtest_results.loc[(episode, 'validation'), 'Number of Trades'] = validation_number_of_trades
    backtest_results.loc[(episode, 'test'), 'Number of Trades'] = test_number_of_trades
    backtest_results.loc[(episode, 'validation'), 'Dataset Index'] = dataset_index
    backtest_results.loc[(episode, 'test'), 'Dataset Index'] = dataset_index

    # Calculate action probabilities for training, validation, and testing datasets
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_train = executor.submit(BF.calculate_probabilities_wrapper, df_train, Trading_Environment_Basic, agent,
                                       look_back, variables, tradable_markets, provision, starting_balance, leverage)
        future_validation = executor.submit(BF.calculate_probabilities_wrapper, df_validation,
                                            Trading_Environment_Basic, agent, look_back, variables, tradable_markets,
                                            provision, starting_balance, leverage)
        future_test = executor.submit(BF.calculate_probabilities_wrapper, df_test, Trading_Environment_Basic, agent,
                                      look_back, variables, tradable_markets, provision, starting_balance, leverage)

        train_probs = future_train.result()
        validation_probs = future_validation.result()
        test_probs = future_test.result()

    episode_probabilities['train'].append(train_probs[['Short', 'Neutral', 'Long']].to_dict(orient='list'))
    episode_probabilities['validation'].append(validation_probs[['Short', 'Neutral', 'Long']].to_dict(orient='list'))
    episode_probabilities['test'].append(test_probs[['Short', 'Neutral', 'Long']].to_dict(orient='list'))

    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(cumulative_reward)
    episode_durations.append(episode_time)
    total_balances.append(env.balance)

    print(
        f"\nCompleted learning from randomly selected window in episode {episode + 1}: Total Reward: {cumulative_reward}, Total Balance: {env.balance:.2f}, Duration: {episode_time:.2f} seconds")
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
        futures.append(executor.submit(BF.process_dataset, df, Trading_Environment_Basic, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage))

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

app.run_server(debug=True, port=8058)

# Open the web browser
webbrowser.open("http://127.0.0.1:8058/")

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
