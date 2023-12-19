# import libraries
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

class BacktestShort:
    def __init__(self, df, look_back=20, variables=None, current_positions=True, tradable_markets='EURUSD',
                 provision=0.0001, agent=None, initial_balance=10000, environment=None):
        # Removed the call to the superclass constructor as it seems unnecessary
        self.df = df.reset_index(drop=True)
        self.df_original = df.copy()
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.current_position = 0
        self.variables = variables
        self.current_positions = current_positions
        self.tradable_markets = tradable_markets
        self.provision = provision
        self.agent = agent
        self.environment = environment

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        if self.current_positions:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.look_back + 1,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.look_back,), dtype=np.float32)

        self.reset()

    def backtest_short(self):
        self.df[('Capital_in', self.tradable_markets)] = 0.0
        self.df[('Capital', 'Strategy')] = 0.0
        self.df[('PnL', self.tradable_markets)] = 0.0
        self.df[('Provision', self.tradable_markets)] = 0.0
        self.df.loc[self.df.index[self.look_back - 1], ('Capital_in', self.tradable_markets)] = self.initial_balance
        self.df.loc[self.df.index[self.look_back - 1], ('Capital', 'Strategy')] = self.initial_balance
        self.df.loc[self.df.index[self.look_back - 1], ('PnL', self.tradable_markets)] = 0.0
        self.df.loc[self.df.index[self.look_back - 1], ('Provision', self.tradable_markets)] = 0.0
        self.df.loc[self.df.index[self.look_back - 1], ('Action', self.tradable_markets)] = 0

        for obs in tqdm(range(len(self.df) - self.look_back)):
            observation = self.environment.reset(obs)
            print(f"Shape of observation in backtest_short: {observation.shape}")
            if obs == 399:
                print(f"Observation in backtest_short: {observation}")
            action = self.agent.choose_best_action(observation)
            self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] = action - 1
            self.df.loc[self.df.index[obs + self.look_back], ('PnL', self.tradable_markets)] = self.df.loc[
                                                                                                   self.df.index[
                                                                                                       obs + self.look_back - 1], (
                                                                                                   'Action',
                                                                                                   self.tradable_markets)] * \
                                                                                               self.df.loc[
                                                                                                   self.df.index[
                                                                                                       obs + self.look_back - 1], (
                                                                                                   'Capital_in',
                                                                                                   self.tradable_markets)] * (
                                                                                                           self.df.loc[
                                                                                                               self.df.index[
                                                                                                                   obs + self.look_back], (
                                                                                                               'Close',
                                                                                                               self.tradable_markets)] /
                                                                                                           self.df.loc[
                                                                                                               self.df.index[
                                                                                                                   obs + self.look_back - 1], (
                                                                                                               'Close',
                                                                                                               self.tradable_markets)] - 1) - \
                                                                                               self.df.loc[
                                                                                                   self.df.index[
                                                                                                       obs + self.look_back], (
                                                                                                   'Provision',
                                                                                                   self.tradable_markets)]
            self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')] = self.df.loc[self.df.index[
                obs + self.look_back - 1], ('Capital', 'Strategy')] + self.df.loc[self.df.index[obs + self.look_back], (
            'PnL', self.tradable_markets)]

            if self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] == self.df.loc[
                self.df.index[obs + self.look_back - 1], ('Action', self.tradable_markets)]:
                self.df.loc[self.df.index[obs + self.look_back], ('Capital_in', self.tradable_markets)] = self.df.loc[
                    self.df.index[obs + self.look_back - 1], ('Capital_in', self.tradable_markets)]

            else:
                if self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] == 0:
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital_in', self.tradable_markets)] = 0
                    self.df.loc[self.df.index[obs + self.look_back], ('Provision', self.tradable_markets)] = \
                    self.df.loc[
                        self.df.index[obs + self.look_back - 1], ('Capital_in', self.tradable_markets)] * self.provision
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')] = self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Capital',
                                                                                                    'Strategy')] - \
                                                                                                self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Provision',
                                                                                                    self.tradable_markets)]

                else:
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital_in', self.tradable_markets)] = \
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')]
                    self.df.loc[self.df.index[obs + self.look_back], ('Provision', self.tradable_markets)] = \
                    self.df.loc[self.df.index[obs + self.look_back - 1], (
                    'Capital_in', self.tradable_markets)] * self.provision * abs(
                        self.df.loc[self.df.index[obs + self.look_back], ('Action', self.tradable_markets)] -
                        self.df.loc[self.df.index[obs + self.look_back - 1], ('Action', self.tradable_markets)])
                    self.df.loc[self.df.index[obs + self.look_back], ('Capital', 'Strategy')] = self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Capital',
                                                                                                    'Strategy')] - \
                                                                                                self.df.loc[
                                                                                                    self.df.index[
                                                                                                        obs + self.look_back], (
                                                                                                    'Provision',
                                                                                                    self.tradable_markets)]
        return self.df

    def calculate_trade_outcomes(self, actions, pnl):
        trade_outcomes = []
        current_trade_pnl = 0
        previous_action = None

        for action, pnl_value in zip(actions, pnl):
            if action != previous_action and action != 0:
                if previous_action is not None:  # End of a trade
                    trade_outcomes.append(current_trade_pnl)
                    current_trade_pnl = 0
            current_trade_pnl += pnl_value
            previous_action = action

        # Add the last trade if it's still open
        if current_trade_pnl != 0:
            trade_outcomes.append(current_trade_pnl)

        return trade_outcomes

    def report_short(self):
        report = {}
        df_log = self.df
        trade_outcomes = self.calculate_trade_outcomes(df_log[('Action', self.tradable_markets)],
                                                       df_log[('PnL', self.tradable_markets)])
        profitable_trades = len([pnl for pnl in trade_outcomes if pnl > 0])
        losing_trades = len([pnl for pnl in trade_outcomes if pnl < 0])
        report['Win Rate (%)'] = (profitable_trades / (losing_trades + profitable_trades)) * 100 if (losing_trades + profitable_trades) != 0 else 0
        report['Total PnL'] = self.df[('PnL', self.tradable_markets)].sum()
        report['Sharpe Ratio'] = self.df[('PnL', self.tradable_markets)].mean() / self.df[('PnL', self.tradable_markets)].std() if self.df[('PnL', self.tradable_markets)].std() != 0 else 0
        report['Total Return'] = self.df[('Capital', 'Strategy')].iloc[-1] / self.initial_balance - 1
        return report

    def plot(self):
        pass

    def reset(self):
        self.df = self.df_original.copy()
        self.current_position = 0

