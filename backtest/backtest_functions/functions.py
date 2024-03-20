"""
# TODO add description
BF - Backtesting Framework
"""
import pandas as pd
import numpy as np
import torch
import math

def generate_predictions_and_backtest(agent_type, df, agent, mkf, look_back, variables, provision=0.001, starting_balance=10000, leverage=1, Trading_Environment_Basic=None, reward_function=None):
    """
    # TODO add description
    """
    action_probabilities_list = []
    best_action_list = []
    balances = []
    number_of_trades = 0

    # Preparing the environment
    if agent_type == 'PPO':
        agent.actor.eval()
        agent.critic.eval()
    elif agent_type == 'DQN':
        agent.q_policy.eval()

    with torch.no_grad():
        # Create a backtesting environment
        env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                        tradable_markets=mkf, provision=provision,
                                        initial_balance=starting_balance, leverage=leverage, reward_function=reward_function)

        observation = env.reset()
        done = False

        while not done:  # TODO check if this is correct
            action_probs = agent.get_action_probabilities(observation, env.current_position)
            best_action = np.argmax(action_probs)

            if (best_action-1) != env.current_position and abs(best_action-1) == 1:
                number_of_trades += 1

            observation_, reward, done, info = env.step(best_action)
            observation = observation_

            balances.append(env.balance)  # Update balances
            action_probabilities_list.append(action_probs.tolist())
            best_action_list.append(best_action-1)

    # KPI Calculations
    returns = pd.Series(balances).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(df)-env.look_back) if returns.std() > 1e-6 else float('nan')

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    negative_volatility = returns[returns < 0].std() * np.sqrt(len(df)-env.look_back)
    sortino_ratio = returns.mean() / negative_volatility if negative_volatility > 1e-6 else float('nan')

    annual_return = cumulative_returns.iloc[-1] ** ((len(df)-env.look_back) / len(returns)) - 1
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else float('nan')

    # Convert the list of action probabilities to a DataFrame
    probabilities_df = pd.DataFrame(action_probabilities_list, columns=['Short', 'Neutral', 'Long'])
    action_df = pd.DataFrame(best_action_list, columns=['Action'])

    # Ensure the agent's networks are reverted back to training mode
    if agent_type == 'PPO':
        agent.actor.train()
        agent.critic.train()
    elif agent_type == 'DQN':
        agent.q_policy.train()

    return env.balance, env.reward_sum, number_of_trades, probabilities_df, action_df, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances

def backtest_wrapper(agent_type, df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic=None, reward_function=None):
    """
    # TODO add description
    """
    return generate_predictions_and_backtest(agent_type, df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic, reward_function)


def make_predictions_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    agent.actor.eval()
    agent.critic.eval()
    with torch.no_grad():
        for observation_idx in range(len(df) - env.look_back):
            observation = env.reset(observation_idx, reset_position=False)
            action = agent.choose_best_action(observation)
            predictions_df.iloc[observation_idx + env.look_back] = action

    df_with_predictions = df.copy()
    df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1
    return df_with_predictions

def make_predictions_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    agent.q_policy.eval()
    with torch.no_grad():
        for observation_idx in range(len(df) - env.look_back):
            observation = env.reset(observation_idx, reset_position=False)
            action = agent.choose_best_action(observation)
            predictions_df.iloc[observation_idx + env.look_back] = action

    df_with_predictions = df.copy()
    df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1
    return df_with_predictions

def calculate_probabilities_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    action_probabilities = []
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    agent.actor.eval()
    agent.critic.eval()
    with torch.no_grad():
        for observation_idx in range(len(df) - env.look_back):
            observation = env.reset(observation_idx, reset_position=False)
            probs = agent.get_action_probabilities(observation)
            action_probabilities.append(probs[0])

    probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Neutral', 'Long'])
    return probabilities_df

def calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    action_probabilities = []
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    agent.q_policy.eval()
    with torch.no_grad():
        for observation_idx in range(len(df) - env.look_back):
            observation = env.reset(observation_idx, reset_position=False)
            probs = agent.get_action_probabilities(observation)
            action_probabilities.append(probs)

    probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Neutral', 'Long'])
    return probabilities_df

def process_dataset_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    predictions = make_predictions_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    probabilities = calculate_probabilities_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    return predictions, probabilities

def process_dataset_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    predictions = make_predictions_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    probabilities = calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    return predictions, probabilities


def calculate_probabilities_wrapper_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    return calculate_probabilities_AC(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)

def calculate_probabilities_wrapper_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    return calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)

def generate_predictions_and_backtest_DQN(df, agent, mkf, look_back, variables, provision=0.0001, initial_balance=10000, leverage=1, reward_scaling=1, Trading_Environment_Basic=None):
    """
    # TODO add description
    # TODO add proper backtest function
    """
    # Switch to evaluation mode
    agent.q_policy.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        df_with_predictions = make_predictions_DQN(df, Trading_Environment_Basic, agent, look_back, variables, mkf, provision, initial_balance, leverage)

        # Backtesting
        balance = initial_balance
        current_position = 0  # Neutral position
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
                if abs(action) == 1:
                    provision_cost = math.log(1 - provision)
                    number_of_trades += 1
                else:
                    provision_cost = 0
            else:
                provision_cost = 0

            reward += provision_cost

            # Update the current position
            current_position = action

            # Update the balance
            balance *= math.exp(reward)

            # Scale reward for better learning
            total_reward += reward * reward_scaling

    # Switch back to training mode
    agent.q_policy.train()

    return balance, total_reward, number_of_trades

def generate_predictions_and_backtest_AC(df, agent, mkf, look_back, variables, provision=0.0001, initial_balance=10000, leverage=1, reward_scaling=1, Trading_Environment_Basic=None):
    """
    # TODO add description
    # TODO add proper backtest function
    AC - Actor Critic
    """
    agent.actor.eval()
    agent.critic.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        df_with_predictions = make_predictions_AC(df, Trading_Environment_Basic, agent, look_back, variables, mkf, provision, initial_balance, leverage)
        # Backtesting
        balance = initial_balance
        current_position = 0  # Neutral position
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
                if abs(action) == 1:
                    provision_cost = math.log(1 - provision)
                    number_of_trades += 1
                else:
                    provision_cost = 0
            else:
                provision_cost = 0

            reward += provision_cost

            # Update the current position
            current_position = action

            # Update the balance
            balance *= math.exp(reward)

            # Scale reward for better learning
            total_reward += reward * reward_scaling

    # Ensure the agent's networks are back in training mode after evaluation
    agent.actor.train()
    agent.critic.train()

    return balance, total_reward, number_of_trades


def backtest_wrapper_AC(df, agent, mkf, look_back, variables, provision, initial_balance, leverage, reward_scaling, Trading_Environment_Basic=None):
    """
    # TODO add description
    AC - Actor Critic
    """
    return generate_predictions_and_backtest_AC(df, agent, mkf, look_back, variables, provision, initial_balance, leverage, reward_scaling, Trading_Environment_Basic)


def backtest_wrapper_DQN(df, agent, mkf, look_back, variables, provision, initial_balance, leverage, reward_scaling, Trading_Environment_Basic=None):
    """
    # TODO add description
    """
    return generate_predictions_and_backtest_DQN(df, agent, mkf, look_back, variables, provision, initial_balance, leverage, reward_scaling, Trading_Environment_Basic)
