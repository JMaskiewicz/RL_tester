import numpy as np
import pandas as pd
from data.function.load_data import load_data
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
import time

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
        '''
        # Check shapes of lists
        # Check shapes of individual elements in the lists
        for idx, state in enumerate(self.states):
            print(f"Shape of state {idx}: {np.array(state).shape}")

        for idx, action in enumerate(self.actions):
            print(f"Shape of action {idx}: {np.array(action).shape}")

        for idx, reward in enumerate(self.rewards):
            print(f"Shape of reward {idx}: {np.array(reward).shape}")

        for idx, done in enumerate(self.dones):
            print(f"Shape of done {idx}: {np.array(done).shape}")

        for idx, prob in enumerate(self.probs):
            print(f"Shape of prob {idx}: {np.array(prob).shape}")

        for idx, val in enumerate(self.vals):
            print(f"Shape of val {idx}: {np.array(val).shape}")
        '''

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

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=16, fc2_dims=16):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.flatten = Flatten()
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=16, fc2_dims=16):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.flatten = Flatten()
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.flatten(x)
        x = self.fc2(x)
        q = self.q(x)

        return q

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=256,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    critic_loss = keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))

        self.memory.clear_memory()

import gym
from gym import spaces

class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=10):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)
        self.look_back = look_back
        self.initial_balance = 10000

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # -1, 0, 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(look_back,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.look_back
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
start_date = '2019-01-01'
split_date = '2020-01-01'
df_train = df[start_date:split_date]
df_test = df[split_date:]

env = Trading_Environment_Basic(df_train)
n_actions = env.action_space.n
agent = Agent(n_actions=n_actions, input_dims=env.observation_space.shape[0])

num_episodes = 10  # Number of episodes for training

total_rewards = []
episode_durations = []

for episode in range(num_episodes):
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
    if episode % 1 == 0:  # Adjust the frequency of printing if needed
        print('Episode: ', episode + 1)
        print("Current Epsilon: ", agent.epsilon)
        print(f"Episode {episode + 1}: Total Reward: {score}, Duration: {episode}, Time: {episode_time:.2f} seconds")
        print('----')


import matplotlib.pyplot as plt
# Plotting the results after all episodes
plt.plot(total_rewards)
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

agent.save_models()