# import libraries
import numpy as np
import random
from collections import deque
from tensorflow.keras import models, layers, optimizers

import pandas as pd
from tqdm import tqdm

from data.function.load_data import load_data
from technical_analysys.add_indicators import add_indicators, compute_volatility

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.995  # exponential decay rate for exploration prob
        self.learning_rate = 0.001  # learning rate
        self.model = self.build_model()
        self.verbose = False
        self.action_space = np.arange(-100, 101, 10)

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        act_values = self.model.predict(state, verbose=self.verbose)
        return self.action_space[np.argmax(act_values[0])]  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=self.verbose)[0]))
            target_f = self.model.predict(state, verbose=self.verbose)
            target_f[0][np.where(self.action_space == action)[0][0]] = target
            self.model.fit(state, target_f, epochs=1, verbose=self.verbose)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Assuming your load_data function and df preprocessing logic is defined correctly
currencies = ['EURUSD']
df = load_data(currencies=currencies, timestamp_x='1d')
df = df.dropna()
start_date = '2022-01-01'
split_date = '2022-09-01'
df_train = df[start_date:split_date]
df_train = df_train[['Close']]

df_test = df[split_date:]
df_test = df_test[['Close']]

episodes = 5
action_size = len(np.arange(-100, 101, 10))
batch_size = 16  # 32
look_back = 10  # Number of previous observations to include
state_size = df_train.shape[1] * look_back
agent = DQNAgent(state_size, action_size)

for e in range(episodes):
    print(f"Episode: {e}/{episodes}")
    total_reward = 0
    for t in tqdm(range(look_back, len(df_train) - 1)):
        state = df_train.iloc[t - look_back:t].values.flatten().reshape(1, -1)
        action = agent.act(state)

        next_state = df_train.iloc[t - look_back + 1:t + 1].values.flatten().reshape(1, -1)
        reward = (df_train.iloc[t + 1]['Close'] - df_train.iloc[t]['Close']) * action
        done = t == len(df_train) - 2

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


# Ensure epsilon is set to 0 for greedy action selection
#agent.epsilon = 0

# TEST
# Initialize an empty list to store the predicted actions
predicted_actions = []

# Loop over the DataFrame to make predictions for each state
for i in range(look_back, len(df_test)):
    # Prepare the state from the current position minus look_back
    current_state = df_test.iloc[i - look_back:i].values.flatten().reshape(1, -1)

    # Predict the action using the trained model
    action = agent.act(current_state)

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
    action = agent.act(current_state)
    predicted_actions.append(action)

padding = [None] * look_back
full_predictions = padding + predicted_actions

df_train['Predicted_Action'] = full_predictions


# After training save model
# agent.save("eurusd-dqn.h5")