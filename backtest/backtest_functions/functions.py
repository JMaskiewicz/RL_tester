"""
# TODO add description
BF - Backtesting Framework
"""
import pandas as pd
import numpy as np
import torch
import math

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
            observation = env.reset(observation_idx)
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
            observation = env.reset(observation_idx)
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
            observation = env.reset(observation_idx)
            probs = agent.get_action_probabilities(observation)
            action_probabilities.append(probs[0])

    probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Neutral', 'Long'])
    df_with_probabilities = df.iloc[env.look_back:].reset_index(drop=True)
    df_with_probabilities = pd.concat([df_with_probabilities, probabilities_df], axis=1)
    return df_with_probabilities

def calculate_probabilities_DQN(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    """
    # TODO add description
    """
    action_probabilities = []
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    agent.q_policy.eval()
    with torch.no_grad():
        for observation_idx in range(len(df) - env.look_back):
            observation = env.reset(observation_idx)
            probs = agent.get_action_probabilities(observation)
            assert probs.shape == (3,), f"Expected probs shape to be (3,), got {probs.shape}"
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
        validation_env = Trading_Environment_Basic(df, look_back=look_back, variables=variables,
                                                   tradable_markets=mkf, provision=provision,
                                                   initial_balance=initial_balance, leverage=leverage, reward_scaling=reward_scaling)

        # Generate Predictions
        predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
        for validation_observation in range(len(df) - validation_env.look_back):
            observation = validation_env.reset()
            action = agent.choose_best_action(observation)  # choose_best_action
            action += - 1  # Convert action to -1, 0, 1
            predictions_df.iloc[validation_observation + validation_env.look_back] = action

        # Merge with original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action']

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
                if abs(action - current_position) == 2:
                    provision_cost = math.log(1 - 2 * provision)
                else:
                    provision_cost = math.log(1 - provision) if action != 0 else 0
                number_of_trades += 1
            else:
                provision_cost = 0

            reward += provision_cost

            # Update the balance
            balance *= math.exp(reward)

            total_reward += reward * reward_scaling  # Scale reward for better learning

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
        env = Trading_Environment_Basic(df, look_back=look_back, variables=variables, tradable_markets=mkf, provision=provision, initial_balance=initial_balance, leverage=leverage, reward_scaling=reward_scaling)

        # Generate Predictions
        predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
        for validation_observation in range(len(df) - env.look_back):
            observation = env.reset()
            action = agent.choose_best_action(observation)  # choose_best_action
            action += - 1  # Convert action to -1, 0, 1
            predictions_df.iloc[validation_observation + env.look_back] = action

        # Merge with original DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action']

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
                if abs(action - current_position) == 2:
                    provision_cost = math.log(1 - 2 * provision)
                else:
                    provision_cost = math.log(1 - provision) if action != 0 else 0
                number_of_trades += 1
            else:
                provision_cost = 0

            reward += provision_cost

            # Update the balance
            balance *= math.exp(reward)

            total_reward += reward * reward_scaling  # Scale reward for better learning

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
