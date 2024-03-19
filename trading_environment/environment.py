import gym
from gym import spaces
import numpy as np
import pandas as pd
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from data.function.edit import normalize_data, standardize_data, process_variable

class Trading_Environment_Basic(gym.Env):
    def __init__(self, df, look_back=20, variables=None, tradable_markets='EURUSD', provision=0.0001,
                 initial_balance=10000, leverage=1, reward_function=None):
        super(Trading_Environment_Basic, self).__init__()
        self.df = df.reset_index(drop=True)  # Reset the index of the DataFrame
        self.look_back = look_back  # Number of time steps to look back
        self.initial_balance = initial_balance  # Initial balance
        self.reward_sum = 0  # Initialize the reward sum
        self.current_position = 0  # This is a static part of the state
        self.variables = variables  # List of variables to be used in the environment
        self.tradable_markets = tradable_markets  # asset to be traded in the environment
        self.provision = provision  # Provision cost
        self.leverage = leverage  # Leverage
        self.balance = initial_balance  # Initialize the balance
        self.reward_function = reward_function

        # Reset the environment to initialize the state
        self.reset()

    def calculate_input_dims(self):
        num_variables = len(self.variables)  # Number of variables
        input_dims = num_variables * self.look_back  # Variables times look_back
        return input_dims

    def reset(self, observation_idx=None, reset_position=True):
        # Reset the environment to the initial state
        if observation_idx is not None:
            self.current_step = observation_idx + self.look_back
        else:
            self.current_step = self.look_back

        # Reset the balance and reward sum
        self.balance = self.initial_balance
        self.reward_sum = 0

        # Reset the current position
        if reset_position:
            self.current_position = 0

        # Set the done flag to False
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        # Get the observation array for the current time step
        start = max(self.current_step - self.look_back, 0)
        end = self.current_step

        # Create a list of the observation arrays for each variable
        tasks = [(self.df[variable['variable']].iloc[start:end].values, variable['edit']) for variable in
                 self.variables]

        # Use ThreadPoolExecutor to parallelize data transformation
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Prepare and execute tasks
            future_to_variable = {executor.submit(process_variable, data, edit_type): (data, edit_type) for data, edit_type in tasks}
            results = []
            for future in concurrent.futures.as_completed(future_to_variable):
                data, edit_type = future_to_variable[future]  # Get the original data and edit type
                try:
                    result = future.result()  # Get the result of the transformation
                except Exception as exc:
                    print('%r generated an exception: %s' % ((data, edit_type), exc))  # Print exception
                else:
                    results.append(result)  # Append the result to the list of results

        # Concatenate results to form the scaled observation array
        scaled_observation = np.concatenate(results).flatten()
        return scaled_observation

    def step(self, action):
        action_mapping = {0: -1, 1: 0, 2: 1}  # best mapping ever :)
        mapped_action = action_mapping[action]  # Map the action to a position change

        # Get the current price and the price of the next time step for the reward calculation and PnL
        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step + 1]

        # balance update
        market_return = next_price / current_price - 1

        # Provision cost calculation if the position has changed
        if mapped_action != self.current_position:
            provision_cost = self.provision * (abs(mapped_action) == 1)
        else:
            provision_cost = 0

        # Update the balance
        self.balance *= (1 + market_return * self.current_position * self.leverage - provision_cost)

        # reward calculation with reward function on the top of the file (reward_calculation)
        final_reward = self.reward_function(current_price, next_price, self.current_position, mapped_action, self.leverage, self.provision)
        self.reward_sum += final_reward  # Update the reward sum
        self.current_position = mapped_action  # Update the current position
        self.current_step += 1  # Increment the current step

        # Check if the episode is done
        self.done = self.current_step >= len(self.df) - 1

        return self._next_observation(), final_reward, self.done, {}

    def simulate_step(self, action, alternative_position):
        action_mapping = {0: -1, 1: 0, 2: 1}
        mapped_action = action_mapping[action]
        alternative_position = action_mapping[alternative_position]

        current_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step]
        next_price = self.df[('Close', self.tradable_markets)].iloc[self.current_step + 1]

        return self.reward_function(current_price, next_price, alternative_position, mapped_action, self.leverage, self.provision)