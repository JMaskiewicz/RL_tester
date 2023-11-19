import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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
        #self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.action_space = np.arange(-100, 101, 25)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 2560),
            nn.ReLU(),
            nn.Linear(2560, 1280),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, self.action_size)
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

# Example usage
currencies = ['EURUSD']
df = load_data(currencies=currencies, timestamp_x='1D')
df = df.dropna()
start_date = '2020-01-01'
split_date = '2022-01-01'
df_train = df[start_date:split_date]
df_train = df_train[['Close']]

df_test = df[split_date:]
df_test = df_test[['Close']]

episodes = 1000
action_size = len(np.arange(-100, 101, 25))
batch_size = 32  # 32
look_back = 20  # Number of previous observations to include
state_size = df_train.shape[1] * look_back
agent = DQNAgent(state_size, action_size)

all_total_rewards = []
episode_durations = []
for e in range(episodes):
    start_time = time.time()
    print(f"Episode: {e}/{episodes}")
    total_reward = 0
    for t in tqdm(range(look_back, len(df_train) - 1)):
        state = df_train.iloc[t - look_back:t].values.flatten().reshape(1, -1)
        action = agent.act(state)

        next_state = df_train.iloc[t - look_back + 1:t + 1].values.flatten().reshape(1, -1)
        reward = (df_train.iloc[t + 1]['Close'] - df_train.iloc[t]['Close']) * action
        done = t == len(df_train) - 2

        agent.remember(state, action, reward, next_state, done)

        # Perform experience replay if enough memory is gathered
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")

    episode_duration = time.time() - start_time  # Calculate duration
    episode_durations.append(episode_duration)
    all_total_rewards.append(total_reward)

# Plotting the results after all episodes
plt.plot(all_total_rewards)
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


# Ensure epsilon is set to 0 for greedy action selection
#agent.epsilon = 0.5
# TEST
# Initialize an empty list to store the predicted actions
predicted_actions = []

# Loop over the DataFrame to make predictions for each state
for i in range(look_back, len(df_test)):
    # Prepare the state from the current position minus look_back
    current_state = df_test.iloc[i - look_back:i].values.flatten().reshape(1, -1)

    # Convert the state to PyTorch tensor and transfer to the device (GPU/CPU)
    current_state_tensor = torch.from_numpy(current_state).float().to(agent.device)

    # Predict the action using the trained model
    action = agent.act(current_state_tensor)

    # Append the predicted action to the list
    predicted_actions.append(action)

# Since we start predicting after 'look_back' rows, we need to prepend NaNs or a placeholder
# to align the length of the predictions with the DataFrame
padding = [None] * look_back
full_predictions = padding + predicted_actions

# Add the predicted actions to the DataFrame as a new column
df_test['Predicted_Action'] = full_predictions

# TRAIN
predicted_actions = []

for i in range(look_back, len(df_train)):
    current_state = df_train.iloc[i - look_back:i].values.flatten().reshape(1, -1)

    # Convert the state to PyTorch tensor and transfer to the device (GPU/CPU)
    current_state_tensor = torch.from_numpy(current_state).float().to(agent.device)

    action = agent.act(current_state_tensor)
    predicted_actions.append(action)

padding = [None] * look_back
full_predictions = padding + predicted_actions

df_train['Predicted_Action'] = full_predictions
