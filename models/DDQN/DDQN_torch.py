import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
from gym import spaces
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data.function.load_data import load_data


# Set seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

class DDQNNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, architecture, l2_reg):
        super(DDQNNetwork, self).__init__()
        layers = []
        for units in architecture:
            layers.append(nn.Linear(state_dim, units))
            layers.append(nn.ReLU())
            state_dim = units  # Update input dimension for the next layer

        layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(architecture[-1], num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DDQNAgent:
    def __init__(self, state_dim, num_actions, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay_steps, epsilon_exponential_decay, replay_capacity, architecture, l2_reg, tau, batch_size, variables):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.batch_size = batch_size
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_network = DDQNNetwork(state_dim, num_actions, architecture, l2_reg)
        self.target_network = DDQNNetwork(state_dim, num_actions, architecture, l2_reg)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.experience = deque(maxlen=replay_capacity)

        self.variables = variables

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(-1, 2)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.online_network(state)
        action = np.argmax(q_values.detach().numpy()) - 1  # Adjusting to new action space
        return action

    def memorize_transition(self, s, a, r, s_prime, not_done):
        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = random.sample(self.experience, self.batch_size)
        states, actions, rewards, next_states, not_done = map(np.array, zip(*minibatch))
        print("states", states.shape)
        print("actions", actions.shape)
        print("rewards", rewards.shape)
        print("next_states", next_states.shape)
        print("not_done", not_done.shape)

        states = torch.FloatTensor(states)
        print("states", states.shape)
        next_states = torch.FloatTensor(next_states)
        print("next_states", next_states.shape)
        actions = torch.LongTensor(actions)
        print("actions", actions.shape)
        rewards = torch.FloatTensor(rewards)
        print("rewards", rewards.shape)
        not_done = torch.FloatTensor(not_done)
        print("not_done", not_done.shape)

        # Compute Q values for current states
        q_values = self.online_network(states)
        print("q_values", q_values.shape)
        # Compute Q values for next states
        next_q_values = self.online_network(next_states)
        print("next_q_values", next_q_values.shape)
        next_q_values_target = self.target_network(next_states)
        print("next_q_values_target", next_q_values_target.shape)

        # Select the best action to take in the next state
        best_actions = torch.argmax(next_q_values, axis=1)
        print("best_actions", best_actions.shape)
        # Gather the Q value corresponding to the best action in the next state
        target_q_values = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        print("target_q_values", target_q_values.shape)
        # Compute the target Q values
        rewards = rewards.unsqueeze(1)
        not_done = not_done.unsqueeze(1)

        targets = rewards + not_done * self.gamma * target_q_values
        print("targets", targets.shape)
        # Update the Q values for the actions taken
        print("q_values shape:", q_values.shape)
        print("actions shape:", actions.shape)
        q_values = q_values.squeeze(1)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        print("current_q_values", current_q_values.shape)
        loss = torch.nn.functional.mse_loss(current_q_values, targets)
        print("loss", loss.shape)
        # Optimize the model
        self.online_network.optimizer.zero_grad()
        loss.backward()
        self.online_network.optimizer.step()

        self.losses.append(loss.item())
        self.total_steps += 1

        # Update the target network
        if self.total_steps % self.tau == 0:
            self.update_target()

        # Epsilon update
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_exponential_decay

    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())


# Example usage
currencies = ['EURUSD']
variables = ['Open', 'High', 'Low', 'Close']
df = load_data(currencies=currencies, timestamp_x='1D')
df = df.dropna()
start_date = '2017-01-01'
split_date = '2019-09-01'
df_train = df[start_date:split_date]
df_test = df[split_date:]

episodes = 30
action_size = len(np.arange(-1, 2, 1))
batch_size = 32  # 32
look_back = 20  # Number of previous observations to include
state_size = df_train.shape[1] * look_back

agent = DDQNAgent(
    state_dim=state_size,
    num_actions=action_size,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=250,
    epsilon_exponential_decay=0.99,
    replay_capacity=int(1e6),
    architecture=(256, 256, 256, 256),
    l2_reg=1e-6,
    tau=100,
    batch_size=4096,
    variables=variables
)

def train_ddqn_agent(agent, df_train, episodes, look_back):
    all_total_rewards = []
    episode_durations = []
    for episode in tqdm(range(episodes)):
        state = get_initial_state(df_train, look_back)
        total_reward = 0
        step = 0

        while True:
            if step >= look_back - 1:  # Ensure complete look-back window
                action = agent.epsilon_greedy_policy(state)
                next_state, reward, done = get_next_state_reward(df_train, state, action, step, look_back)
                agent.memorize_transition(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                agent.experience_replay()

                if step % agent.tau == 0:
                    agent.update_target()

                if done:
                    break

            step += 1

        all_total_rewards.append(total_reward)
        episode_durations.append(step)
        if episode % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Duration: {step}")

    return all_total_rewards, episode_durations


def get_initial_state(df, look_back):
    initial_data = df.iloc[:look_back][variables].values.flatten().reshape(1, -1)
    return initial_data

def get_next_state_reward(df, state, action, t, look_back):
    start_index = max(0, t + 1 - look_back)
    next_state = df.iloc[start_index: t + 1][variables].values.flatten()

    # Padding if necessary
    if len(next_state) < look_back:
        padding = np.zeros(look_back - len(next_state))
        next_state = np.concatenate((padding, next_state), axis=0)

    # Reshape next_state to ensure it's always (1, look_back)
    next_state = next_state.reshape(1, -1)  # Reshape to (1, look_back)

    # Calculate reward as action * difference in close prices
    if t + 1 < len(df) - 1:
        price_diff = df.iloc[t + 1]['Close'] - df.iloc[t]['Close']
        reward = action * price_diff.iloc[0]
    else:
        reward = 0

    done = t == len(df) - 1
    return next_state, reward, done

all_total_rewards, episode_durations = train_ddqn_agent(agent, df_train, episodes, look_back)

# Plotting the results after all episodes
plt.plot(all_total_rewards)
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


def make_predictions_with_ddqn_agent(agent, df, look_back):
    state = get_initial_state(df, look_back)
    predicted_actions = []
    rewards = []
    total_reward = 0

    for t in range(len(df) - look_back):
        action = agent.epsilon_greedy_policy(state)
        next_state, reward, done = get_next_state_reward(df, state, action, t, look_back)

        predicted_actions.append(action)
        rewards.append(reward)
        total_reward += reward

        state = next_state
        if done:
            break

    return predicted_actions, rewards, total_reward


#predicted_actions, rewards, total_reward = make_predictions_with_ddqn_agent(agent, df_train, look_back)
#print(f"Total Reward: {total_reward}")