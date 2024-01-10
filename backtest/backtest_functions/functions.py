import pandas as pd
import numpy as np

def make_predictions(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    predictions_df = pd.DataFrame(index=df.index, columns=['Predicted_Action'])
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    for observation_idx in range(len(df) - env.look_back):
        observation = env.reset(observation_idx)
        action = agent.choose_best_action(observation)
        predictions_df.iloc[observation_idx + env.look_back] = action

    df_with_predictions = df.copy()
    df_with_predictions['Predicted_Action'] = predictions_df['Predicted_Action'] - 1
    return df_with_predictions


def calculate_probabilities(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    action_probabilities = []
    env = environment_class(df, look_back=look_back, variables=variables, tradable_markets=tradable_markets, provision=provision, initial_balance=starting_balance, leverage=leverage)

    for observation_idx in range(len(df) - env.look_back):
        observation = env.reset(observation_idx)
        probs = agent.get_action_probabilities(observation)
        action_probabilities.append(probs[0])

    probabilities_df = pd.DataFrame(action_probabilities, columns=['Short', 'Do_nothing', 'Long'])
    df_with_probabilities = df.iloc[env.look_back:].reset_index(drop=True)
    df_with_probabilities = pd.concat([df_with_probabilities, probabilities_df], axis=1)
    return df_with_probabilities


def process_dataset(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    predictions = make_predictions(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    probabilities = calculate_probabilities(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)
    return predictions, probabilities

def calculate_probabilities_wrapper(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage):
    return calculate_probabilities(df, environment_class, agent, look_back, variables, tradable_markets, provision, starting_balance, leverage)