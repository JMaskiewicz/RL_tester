"""
Second version of Proximal Policy Optimization (PPO) with Long Short-Term Memory (LSTM) network
- reward is the key to learning, if the reward is too close to 0 (reward when agent decide to be out of market), the agent will not learn
-
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
import torch.nn.functional as F
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


# TODO add proper backtest function
def generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision=0.0001, initial_balance=10000,
                                      leverage=1):
    # Create a validation environment
    validation_env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                               tradable_markets=tradable_markets, provision=provision,
                                               initial_balance=initial_balance, leverage=leverage)

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
            else:
                provision_cost = math.log(1 - provision) if action != 0 else 0
            number_of_trades += 1
        else:
            provision_cost = 0

        reward += provision_cost

        # Update the balance
        balance *= math.exp(reward)

        # Scale the reward
        scaled_reward = reward * 1000
        total_reward += scaled_reward  # Accumulate total reward

        # Update current position
        current_position = action

    return balance, total_reward, number_of_trades

def backtest_wrapper(df, agent, mkf, look_back, variables, provision, initial_balance, leverage):
    return generate_predictions_and_backtest(df, agent, mkf, look_back, variables, provision, initial_balance, leverage)


# TODO use deque instead of list
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.static_inputs = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        states = np.array(self.states)[indices]
        actions = np.array(self.actions)[indices]
        probs = np.array(self.probs)[indices]
        vals = np.array(self.vals)[indices]
        rewards = np.array(self.rewards)[indices]
        dones = np.array(self.dones)[indices]
        static_inputs = np.array(self.static_inputs)[indices]

        return states, actions, probs, vals, rewards, dones, static_inputs, batches

    def store_memory(self, state, action, probs, vals, reward, done, static_input):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(float(reward))
        self.dones.append(done)
        self.static_inputs.append(static_input)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.static_inputs = []
class LSTM_NetworkBase(nn.Module):
    def __init__(self, input_dims, static_dim, hidden_size=1024, n_layers=2):
        super(LSTM_NetworkBase, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dims, hidden_size=hidden_size,
                            batch_first=True, num_layers=n_layers, dropout=0.2)

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
        return combined_input

class LSTM_ActorNetwork(LSTM_NetworkBase):
    def __init__(self, n_actions, input_dims, static_dim, hidden_size=2048):
        super(LSTM_ActorNetwork, self).__init__(input_dims, static_dim, hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 256),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(1/16)
        )
        self.policy = nn.Linear(128, n_actions)

    def forward(self, state, static_input):
        combined_input = super().forward(state, static_input).squeeze(0)
        x = self.fc1(combined_input)
        action_probs = torch.softmax(self.policy(x), dim=-1)
        return action_probs

class LSTM_CriticNetwork(LSTM_NetworkBase):
    def __init__(self, input_dims, static_dim, hidden_size=2048):
        super(LSTM_CriticNetwork, self).__init__(input_dims, static_dim, hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 256),
            nn.ReLU(),
            nn.Dropout(1/16),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(1/16)
        )
        self.value = nn.Linear(128, 1)

    def forward(self, state, static_input):
        combined_input = super().forward(state, static_input).squeeze(0)
        x = self.fc1(combined_input)
        value = self.value(x)
        return value

class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.001, gae_lambda=0.9, policy_clip=0.1, batch_size=1024, n_epochs=20, mini_batch_size=128, entropy_coefficient=0.01, weight_decay=0.0001, l1_lambda=1e-5):
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

        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done, static_input):
        self.memory.store_memory(state, action, probs, vals, reward, done, static_input)

    def learn(self):
        for _ in range(self.n_epochs):
            # Generating the data for the entire batch
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, static_input_arr, batches = self.memory.generate_batches()

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
                static_input_batch = static_input_arr[minibatch_indices]

                # Extract data for the current mini-batch
                batch_states = torch.tensor(state_arr[minibatch_indices], dtype=torch.float).to(self.device)
                batch_actions = torch.tensor(action_arr[minibatch_indices], dtype=torch.long).to(self.device)
                batch_old_probs = torch.tensor(old_prob_arr[minibatch_indices], dtype=torch.float).to(self.device)
                batch_advantages = advantage[minibatch_indices]
                batch_values = values[minibatch_indices]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Actor Network Loss with Entropy Regularization
                probs = self.actor(batch_states, torch.tensor(static_input_batch, dtype=torch.float).to(self.device))
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
                critic_value = self.critic(batch_states, torch.tensor(static_input_batch, dtype=torch.float).to(self.device)).squeeze()
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

    def get_action_probabilities(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        observation = observation.reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        static_input = torch.tensor([0], dtype=torch.float).to(self.device)

        # Get the action probabilities from the actor network
        action_probs = self.actor(state, static_input).cpu().detach().numpy()
        return action_probs

    def choose_best_action(self, observation):
        # Get action probabilities
        action_probs = self.get_action_probabilities(observation)
        # Choose the action with the highest probability
        best_action = np.argmax(action_probs)
        return best_action


class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001, initial_balance=10000, leverage=1, window_size_statics=30):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables

        self.tradable_markets = tradable_markets
        self.provision = provision
        self.leverage = leverage

        # for reward function
        self.holding_duration = 0
        self.rolling_returns = []
        self.rolling_downside_returns = []
        self.window_size_statics = window_size_statics

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0 (sell), 1 (hold), 2 (buy)

        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        return input_dims

    def reset(self, observation=None):
        if observation is not None:
            self.current_step = observation + self.look_back
        else:
            self.current_step = self.look_back

        self.balance = self.initial_balance
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
            else:
                scaled_data = data

            scaled_observation.extend(scaled_data)
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
        reward = reward * self.leverage

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
        final_reward = reward * 1000

        # Check if the episode is done
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._next_observation(), final_reward, self.done, {}

    # TODO
    def render(self):
        pass
if __name__ == '__main__':
    # Example usage
    # Stock market variables
    df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY'], '1D')

    indicators = [
        {"indicator": "RSI", "mkf": "EURUSD", "length": 14},
        {"indicator": "ATR", "mkf": "EURUSD", "length": 24},]

    add_indicators(df, indicators)

    df[("RSI_14", "EURUSD")] = df[("RSI_14", "EURUSD")]/100
    df = df.dropna()
    start_date = '2013-01-01'
    validation_date = '2021-01-01'
    test_date = '2022-01-01'
    df_train = df[start_date:validation_date]
    df_validation = df[validation_date:test_date]
    df_test = df[test_date:]
    variables = [
        {"variable": ("Close", "EURUSD"), "edit": "normalize"},
    ]
    tradable_markets = 'EURUSD'
    window_size = '1Y'
    starting_balance = 10000
    look_back = 10
    provision = 0.01  # 0.001, cant be too high as it would not learn to trade

    # Training parameters
    batch_size = 256
    epochs = 1
    mini_batch_size = 32
    leverage = 1
    weight_decay = 0.005
    l1_lambda = 1e-5
    # Create the environment
    env = Trading_Environment_Basic(df_train, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    agent = PPO_Agent(n_actions=env.action_space.n,
                      input_dims=env.calculate_input_dims(),
                      gamma=0.9,
                      alpha=0.01,  # lower learning rate
                      gae_lambda=0.9,
                      policy_clip=0.2,
                      entropy_coefficient=0.1,  # maybe try higher entropy coefficient
                      batch_size=batch_size,
                      n_epochs=epochs,
                      mini_batch_size=mini_batch_size,
                      weight_decay=weight_decay,
                      l1_lambda=l1_lambda)

    num_episodes = 10  # 250

    total_rewards = []
    episode_durations = []
    total_balances = []
    episode_probabilities = {'train': [], 'validation': [], 'test': []}

    # Assuming df_train is your DataFrame
    rolling_datasets = rolling_window_datasets(df_train, window_size=window_size,  look_back=look_back)

    # Create a DataFrame to store the backtest results
    index = pd.MultiIndex.from_product([range(num_episodes), ['validation', 'test']], names=['episode', 'dataset'])
    columns = ['Final Balance', 'Dataset Index']  # Add 'Dataset Index' column
    backtest_results = pd.DataFrame(index=index, columns=columns)

    # Use 'cycle' to endlessly iterate over the rolling_datasets
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
            current_position = env.current_position
            action, prob, val = agent.choose_action(observation, current_position)
            static_input = current_position
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, prob, val, reward, done, static_input)
            observation = observation_
            cumulative_reward += reward

            # Check if enough data is collected or if the dataset ends
            if len(agent.memory.states) >= agent.memory.batch_size or done:
                agent.learn()
                agent.memory.clear_memory()


        # results
        end_time = time.time()
        episode_time = end_time - start_time
        total_rewards.append(cumulative_reward)
        episode_durations.append(episode_time)
        total_balances.append(env.balance)

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

    # ploting probabilities
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
        dcc.Graph(id='probability-plot')
    ])

    # Callback to update graph
    @app.callback(
        Output('probability-plot', 'figure'),
        [Input('dataset-dropdown', 'value'), Input('episode-input', 'value')]
    )
    def update_graph(selected_dataset, selected_episode):
        data = episode_probabilities[selected_dataset][selected_episode]
        x_values = list(range(len(data['Short'])))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=data['Short'], mode='lines', name='Short'))
        fig.add_trace(go.Scatter(x=x_values, y=data['Neutral'], mode='lines', name='Neutral'))
        fig.add_trace(go.Scatter(x=x_values, y=data['Long'], mode='lines', name='Long'))

        fig.update_layout(title='Action Probabilities Over Episodes',
                          xaxis_title='Time',
                          yaxis_title='Probability')
        return fig

    app.run_server(debug=True)

    import webbrowser

    # Open the web browser
    webbrowser.open("http://127.0.0.1:8050/")

    # http://127.0.0.1:8050/

    print('end')