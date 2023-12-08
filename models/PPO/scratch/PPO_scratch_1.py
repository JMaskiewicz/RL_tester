"""
PPO agent version 1
Adding transaction costs to the environment ie opening and closing positions

"""


import numpy as np
import pandas as pd
from data.function.load_data import load_data
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
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
    def __init__(self, n_actions, input_dims):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, n_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        q = self.fc4(x)
        return q

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0005, gae_lambda=0.95, policy_clip=0.2, batch_size=1024, n_epochs=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims).to(self.device)
        self.critic = CriticNetwork(input_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        # Ensure observation is a numpy array and reshape to add a batch dimension
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

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

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

            for batch in batches:
                # Ensure each state in the batch is correctly reshaped
                batch_states = np.array([state.reshape(-1) for state in state_arr[batch]])
                states = torch.tensor(batch_states, dtype=torch.float).to(self.device)

                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float).to(self.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.long).to(self.device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Actor Network Loss
                probs = self.actor(states)
                dist = torch.distributions.Categorical(probs)
                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                weighted_clipped_probs = clipped_probs * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Critic Network Loss
                critic_value = self.critic(states).squeeze()
                returns = advantage[batch] + values[batch]
                critic_loss = nn.functional.mse_loss(critic_value, returns)

                # Gradient Calculation and Optimization Step
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

    def get_action_probabilities(self, observation):
        """
        Returns the probabilities of each action for a given observation.
        """
        observation = np.array(observation).reshape(1, -1)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        with torch.no_grad():
            probs = self.actor(state)
        return probs.cpu().numpy()

import gym
from gym import spaces

class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = 10000

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # -1, 0, 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(look_back,), dtype=np.float32)

        self.reset()

    def reset(self, day=None):
        if day is not None:
            self.current_step = day + self.look_back
        else:
            self.current_step = self.look_back

        self.balance = self.initial_balance
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        return self.df['Close'].iloc[self.current_step-self.look_back:self.current_step].values

    def step(self, action):
        # Map the action from 0, 1, 2 to -1, 0, 1
        action_mapping = {0: -1, 1: 0, 2: 1}
        mapped_action = action_mapping[action]

        # Calculate reward and update balance
        current_price = self.df['Close'].iloc[self.current_step]
        next_price = self.df['Close'].iloc[self.current_step + 1]

        reward = 0
        if mapped_action == 1:  # Buying
            reward = (next_price - current_price) / current_price * self.balance
            self.balance += reward
        elif mapped_action == -1:  # Selling
            reward = (current_price - next_price) / current_price * self.balance
            self.balance += reward
        else:
            reward = 0
            self.balance += reward

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._next_observation(), reward, self.done, {}

    # TODO
    def render(self):
        pass


# Example usage
currencies = ['EURUSD']
df = load_data(currencies=currencies, timestamp_x='1D')
df = df.dropna()
start_date = '2015-01-01'
split_date = '2020-01-01'
df_train = df[start_date:split_date]
df_test = df[split_date:]
df_test_probs = df_test.copy()
print('number of train samples: ', len(df_train))
print('number of test samples: ', len(df_test))

env = Trading_Environment_Basic(df_train)
n_actions = env.action_space.n
agent = Agent(n_actions=n_actions, input_dims=env.observation_space.shape[0], batch_size=512)

num_episodes = 40  # Number of episodes for training

total_rewards = []
episode_durations = []

for episode in tqdm(range(num_episodes)):
    observation = env.reset()
    done = False
    score = 0
    start_time = time.time()
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, prob, val, reward, done)
        observation = observation_
        score += reward

        if len(agent.memory.states) == agent.memory.batch_size:
            agent.learn()

    end_time = time.time()
    episode_time = end_time - start_time
    total_rewards.append(score)
    episode_durations.append(episode_time)
    if episode % 1 == 0:
        print('Episode: ', episode + 1)
        print(f"Episode {episode + 1}: Total Reward: {score}, Duration: {episode}, Time: {episode_time:.2f} seconds")
        print('----\n')


import matplotlib.pyplot as plt
# Plotting the results after all episodes
plt.plot(total_rewards)
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


# final prediction
predictions_df = pd.DataFrame(index=df_test.index, columns=['Predicted_Action'])
test_env = Trading_Environment_Basic(df_test)

for test_day in range(len(df_test) - test_env.look_back):
    observation = test_env.reset(test_day)  # Reset environment to the specific day
    action, _, _ = agent.choose_action(observation)

    # print(f'Day: {test_day + test_env.look_back}, Chosen Action: {action}')
    predictions_df.iloc[test_day + test_env.look_back] = action

# Merge with df_test
df_test_with_predictions = df_test.copy()
df_test_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action']-1

print(df_test_with_predictions)

# final prediction with probabilities
test_env_probs = Trading_Environment_Basic(df_test_probs)
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

# final prediction with probabilities
df_train_probs = df_train.copy()
train_env_probs = Trading_Environment_Basic(df_train_probs)
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
