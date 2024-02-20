import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from tqdm import tqdm

from data.function.load_data import load_data


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.action_space = np.arange(-100, 101, 10)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.float().to(self.device)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        return self.action_space[torch.argmax(act_values[0]).item()]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.tensor(reward).float().to(self.device)
            action_index = np.where(self.action_space == action)[0][0]

            self.optimizer.zero_grad()
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state)
            target_f[0][action_index] = target
            loss = nn.MSELoss()(target_f, target_f.clone().detach())
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def run_on_device(device_type):
    start_time = time.time()  # Start timing
    device = torch.device(device_type)

    currencies = ['EURUSD']
    df = load_data(currencies=currencies, timestamp_x='1d')
    df = df.dropna()
    start_date = '2010-01-01'
    split_date = '2021-01-01'
    df_train = df[start_date:split_date]
    df_train = df_train[['Close']]

    df_test = df[split_date:]
    df_test = df_test[['Close']]

    episodes = 10
    action_size = len(np.arange(-100, 101, 10))
    batch_size = 32  # 32
    look_back = 10  # Number of previous observations to include
    state_size = df_train.shape[1] * look_back
    agent = DQNAgent(state_size, action_size)

    state_size = df_train.shape[1] * look_back
    action_size = len(np.arange(-100, 101, 10))

    agent = DQNAgent(state_size, action_size)
    agent.device = device
    agent.model.to(device)

    # Training loop
    for e in range(episodes):
        print(f"Episode: {e}/{episodes}")
        total_reward = 0
        for t in tqdm(range(look_back, len(df_train) - 1)):
            state = df_train.iloc[t - look_back:t].values.flatten().reshape(1, -1)
            state_tensor = torch.from_numpy(state).float().to(device)

            action = agent.act(state_tensor)

            next_state = df_train.iloc[t - look_back + 1:t + 1].values.flatten().reshape(1, -1)
            next_state_tensor = torch.from_numpy(next_state).float().to(device)

            reward = (df_train.iloc[t + 1]['Close'] - df_train.iloc[t]['Close']) * action
            done = t == len(df_train) - 2

            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)

            total_reward += reward
            if done:
                print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")


    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Training completed in: {total_time:.2f} seconds")

# To run on GPU (if available)
if torch.cuda.is_available():
    run_on_device("cuda")
else:
    print("CUDA is not available. Running on CPU instead.")

# Example usage
run_on_device("cpu")

