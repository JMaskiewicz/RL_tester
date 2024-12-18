import numpy as np

class Buy_and_hold_Agent:
    def __init__(self, action_size=3):
        self.action_size = action_size
        self.generation = 'Buy and Hold'

    def get_action_probabilities(self, observation, current_position):
        action_probs = np.zeros(self.action_size)
        action_probs[2] = 1.0
        return action_probs

class Sell_and_hold_Agent:
    def __init__(self, action_size=3):
        self.action_size = action_size
        self.generation = 'Sell and Hold'

    def get_action_probabilities(self, observation, current_position):
        action_probs = np.zeros(self.action_size)
        action_probs[0] = 1.0
        return action_probs

class Yearly_Perfect_Agent:
    def __init__(self, df, look_back, tradable_markets, action_size=3):
        self.df = df
        self.look_back = look_back
        self.action_size = action_size
        self.tradable_markets = tradable_markets
        self.generation = 'Yearly Perfect Agent'
        self.trends = self.analyze_market_trends()
        self.precomputed_actions = self.precompute_actions()
        self.current_step = 0

    def analyze_market_trends(self):
        trends = {}
        for year in self.df.index.year.unique():
            year_data = self.df.loc[self.df.index.year == year]
            initial_price = year_data[('Close', self.tradable_markets)].iloc[0]
            final_price = year_data[('Close', self.tradable_markets)].iloc[-1]
            market_return = final_price / initial_price - 1
            trends[year] = 1 if market_return > 0 else -1
        return trends  # dict(list(trends.items())[self.look_back:])  # TODO CORRECT THIS

    def precompute_actions(self):
        actions = []
        for idx in range(len(self.df)):
            current_year = self.df.index[idx].year
            trend = self.trends[current_year]
            if trend == 1:
                actions.append(2)  # Long position
            else:
                actions.append(0)  # Short position
        return actions

    def get_action_probabilities(self, observation, current_position):
        action_probs = np.zeros(self.action_size)
        precomputed_action = self.precomputed_actions[self.current_step]
        action_probs[precomputed_action] = 1.0
        self.current_step += 1
        return action_probs


# tests
if __name__ == '__main__':
    # test the agents
    from trading_environment.environment import Trading_Environment_Basic
    import numpy as np
    import pandas as pd
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import random
    from itertools import cycle
    from concurrent.futures import ThreadPoolExecutor
    import torch.nn.functional as F
    from numba import jit
    from torch.optim.lr_scheduler import ExponentialLR
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    from data.function.load_data import load_data_parallel
    from data.function.rolling_window import rolling_window_datasets
    from technical_analysys.add_indicators import add_indicators, add_returns, add_log_returns, add_time_sine_cosine
    from functions.utilis import save_model
    import backtest.backtest_functions.functions as BF
    from functions.utilis import prepare_backtest_results, generate_index_labels, get_time

    @jit(nopython=True)
    def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
        # Calculate the normal return
        if previous_close != 0:
            normal_return = (current_close - previous_close) / previous_close
        else:
            normal_return = 0

        # Calculate the base reward
        reward = normal_return * current_position * 1000

        # Penalize the agent for taking the wrong action
        if reward < 0:
            reward *= 1  # penalty for wrong action

        # Calculate the cost of provision if the position has changed, and it's not neutral (0).
        if current_position != previous_position and abs(current_position) == 1:
            provision_cost = - provision * 1000  # penalty for changing position
        elif current_position == previous_position and abs(current_position) == 1:
            provision_cost = + provision * 0
        else:
            provision_cost = 0

        # Apply the provision cost
        reward += provision_cost

        # Scale the reward to enhance its significance for the learning process
        final_reward = reward

        return final_reward

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    tradable_markets = 'USDJPY'
    starting_balance = 10000
    look_back = 1
    provision = 0.0001
    leverage = 1

    # final results for the agent
    # Example usage
    df = load_data_parallel(['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD'], '1D')

    df = df.dropna()

    start_date = '2005-01-01'  # worth to keep 2008 as it was a financial crisis
    validation_date = '2017-12-31'
    test_date = '2018-12-31'

    df_train, df_validation, df_test = df[start_date:validation_date], df[validation_date:test_date], df[
                                                                                   test_date:'2024-01-01']
    #df_validation = pd.concat([df_train.iloc[-look_back:], df_validation])
    #df_test = pd.concat([df_validation.iloc[-look_back:], df_test])

    variables = [
        {"variable": ("Close", tradable_markets), "edit": "standardize"},
    ]

    buy_and_hold_agent = Buy_and_hold_Agent()
    sell_and_hold_agent = Sell_and_hold_Agent()

    # Run backtesting for both agents
    bah_results, _, benchmark_BAH = BF.run_backtesting(
        buy_and_hold_agent, 'BAH', [df_test], ['final_test'],
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, starting_balance, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    sah_results, _, benchmark_SAH = BF.run_backtesting(
        sell_and_hold_agent, 'SAH', [df_test], ['final_test'],
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, starting_balance, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    bah_results_prepared = prepare_backtest_results(bah_results, 'BAH')
    sah_results_prepared = prepare_backtest_results(sah_results, 'SAH')

    sah_results_prepared = sah_results_prepared.drop(('', 'Agent Generation'),
                                                     axis=1)  # drop the agent generation column
    bah_results_prepared = bah_results_prepared.drop(('', 'Agent Generation'),
                                                     axis=1)  # drop the agent generation column

    perfect_agent = Yearly_Perfect_Agent(df_test, look_back, tradable_markets, action_size=3)  # PH - perfect hold

    # Run backtesting for both agents
    ph_results, _, benchmark_ph = BF.run_backtesting(
        perfect_agent, 'PH', [df_test], ['test'],
        BF.backtest_wrapper, tradable_markets, look_back, variables, provision, starting_balance, leverage,
        Trading_Environment_Basic, reward_calculation, workers=4)

    ph_results_prepared = prepare_backtest_results(ph_results, 'PH')

    print("Buy and Hold Agent final results:", bah_results_prepared[('BAH', 'Final Balance')][0])
    print("Benchmark Buy and Hold Agent final reward:", bah_results_prepared[('BAH', 'Total Reward')][0])
    print("Sell and Hold Agent final results:", sah_results_prepared[('SAH', 'Final Balance')][0])
    print("Benchmark Sell and Hold Agent final reward:", sah_results_prepared[('SAH', 'Total Reward')][0])
    print("Perfect Hold Agent final results:", ph_results_prepared[('PH', 'Final Balance')][0])
    print("Benchmark Perfect Hold Agent final reward:", ph_results_prepared[('PH', 'Total Reward')][0])

    first_close = df_test.loc[:, ('Close', tradable_markets)].iloc[look_back]
    last_close = df_test.loc[:, ('Close', tradable_markets)].iloc[-1]
    return_percentage = ((last_close - first_close) / first_close) * 100

    # Calculate the final investment value
    final_investment_value = (1 + (return_percentage / 100)) * starting_balance

    # Output the final value
    print("Final Investment Value:", final_investment_value)

    for column in bah_results_prepared.columns:
        if len(bah_results_prepared[column]) > 0:
            print(f"{column}: {bah_results_prepared[column][0]}")

    for column in sah_results_prepared.columns:
        if len(sah_results_prepared[column]) > 0:
            print(f"{column}: {sah_results_prepared[column][0]}")

    for column in ph_results_prepared.columns:
        if len(ph_results_prepared[column]) > 0:
            print(f"{column}: {ph_results_prepared[column][0]}")

