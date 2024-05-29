"""
# TODO add description
BF - Backtesting Framework
"""
import pandas as pd
import numpy as np
import torch
import math
from concurrent.futures import ThreadPoolExecutor

#from functions.utilis import prepare_backtest_results, generate_index_labels, get_time

def calculate_drawdown_duration(drawdown):
    # Identifying the drawdown periods
    is_drawdown = drawdown < 0

    # Start a new group every time there is a change from drawdown to non-drawdown
    drawdown_groups = is_drawdown.ne(is_drawdown.shift()).cumsum()

    # Filter out only the drawdown periods
    drawdown_periods = drawdown_groups[is_drawdown]

    # Count consecutive indexes in each drawdown period
    drawdown_durations = drawdown_periods.groupby(drawdown_periods).transform('count')

    # The longest duration of drawdown
    max_drawdown_duration = drawdown_durations.max()

    return max_drawdown_duration

#@get_time
def run_backtesting(agent, agent_type, datasets, labels, backtest_wrapper, currency_pair, look_back,
                    variables, provision, starting_balance, leverage, Trading_Environment_Class, reward_calculation,
                    workers=4):
    backtest_results = {}
    probs_dfs = {}
    balances_dfs = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for df, label in zip(datasets, labels):
            # Pass parameters properly to the backtest_wrapper function.
            future = executor.submit(backtest_wrapper, agent_type, df, agent, currency_pair, look_back,
                                     variables, provision, starting_balance, leverage,
                                     Trading_Environment_Class, reward_calculation)
            futures.append((future, label))

        for future, label in futures:
            result = future.result()
            balance, total_reward, number_of_trades, probabilities_df, action_df, sharpe_ratio, max_drawdown, \
            sortino_ratio, calmar_ratio, cumulative_returns, balances, provision_sum, max_drawdown_duration, \
            average_trade_duration, in_long, in_short, in_out_of_market, win_rate, annualised_returns, \
            annualised_std = result

            # Update result_data with all required metrics
            result_data = {
                'Agent generation': agent.generation,
                'Agent Type': agent_type,
                'Label': label,
                'Provision_sum': provision_sum,
                'Final Balance': balance,
                'Total Reward': total_reward,
                'Number of Trades': number_of_trades,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Max Drawdown Duration': max_drawdown_duration,
                'Average Trade Duration': average_trade_duration,
                'In Long': in_long,
                'In Short': in_short,
                'In Out of the Market': in_out_of_market,
                'Win Rate': win_rate,
                'Average Yearly Return': annualised_returns,
                'Average Yearly std': annualised_std,
            }

            key = (agent.generation, label)
            backtest_results.setdefault(key, []).append(result_data)
            probs_dfs[key] = probabilities_df
            balances_dfs[key] = balances

    return backtest_results, probs_dfs, balances_dfs


def generate_predictions_and_backtest(agent_type, df, agent, mkf, look_back, variables, provision=0.001,
                                      starting_balance=10000, leverage=1, Trading_Environment_Basic=None,
                                      reward_function=None, annualization_factor=365, risk_free_rate=0.0):
    """
    Generate predictions and backtest a trading agent.

    Parameters:
        agent_type (str): The type of agent (e.g., 'PPO', 'DQN').
        df (pd.DataFrame): The dataframe containing market data.
        agent: The trading agent.
        mkf (list): List of market features.
        look_back (int): The look-back period for the environment.
        variables (list): List of variables to use in the environment.
        provision (float): The trading provision or fee.
        starting_balance (float): The starting balance for the backtest.
        leverage (float): The leverage for the backtest.
        Trading_Environment_Basic (class): The trading environment class.
        reward_function (function): The reward function for the environment.
        annualization_factor (int): The factor for annualizing metrics.

    Returns:
        tuple: A tuple containing various backtest results and statistics.
    """
    action_probabilities_list = []
    best_action_list = []
    balances = []

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
                                        initial_balance=starting_balance, leverage=leverage,
                                        reward_function=reward_function)

        observation = env.reset()
        done = False

        while not done:  # TODO check if this is correct
            action_probs = agent.get_action_probabilities(observation, env.current_position)
            best_action = np.argmax(action_probs)
            observation_, reward, done, info = env.step(best_action)
            observation = observation_

            balances.append(env.balance)  # Update balances
            action_probabilities_list.append(action_probs.tolist())
            best_action_list.append(best_action-1)

    # KPI Calculations
    returns = pd.Series([starting_balance] + balances).pct_change().dropna()
    annual_return = (pd.Series(balances).iloc[-1] / starting_balance) ** (annualization_factor / len(returns)) - 1
    annual_std = returns.std() * np.sqrt(annualization_factor)
    sharpe_ratio = annual_return / annual_std if returns.std() > 1e-6 else float('nan')

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    max_drawdown_duration = calculate_drawdown_duration(drawdown)

    negative_volatility = returns[returns < 0].std() * np.sqrt(annualization_factor)
    sortino_ratio = annual_return / negative_volatility if negative_volatility > 1e-6 else float('nan')

    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else float('nan')

    # Convert the list of action probabilities to a DataFrame
    probabilities_df = pd.DataFrame(action_probabilities_list, columns=['Short', 'Neutral', 'Long'])
    action_df = pd.DataFrame(best_action_list, columns=['Action'])

    action_df['Action'] = action_df['Action'].map({-1: 'Short', 0: 'Neutral', 1: 'Long'})

    # Pass the 'Action' column Series directly
    num_trades, average_trade_duration = calculate_number_of_trades_and_duration(action_df['Action'])

    # Calculate the number of times the agent was in long, short, or out of the market
    in_long = action_df[action_df['Action'] == 'Long'].shape[0] / (len(df) - env.look_back - 1)
    in_short = action_df[action_df['Action'] == 'Short'].shape[0] / (len(df) - env.look_back - 1)
    in_out_of_market = action_df[action_df['Action'] == 'Neutral'].shape[0] / (len(df) - env.look_back - 1)

    # Ensure the agent's networks are reverted back to training mode
    if agent_type == 'PPO':
        agent.actor.train()
        agent.critic.train()
    elif agent_type == 'DQN':
        agent.q_policy.train()

    win_rate = env.profitable_trades / env.num_trades if env.num_trades > 0 else 0

    return (env.balance, env.reward_sum, env.num_trades, probabilities_df, action_df, sharpe_ratio, max_drawdown,  # 7
            sortino_ratio, calmar_ratio, cumulative_returns, balances, env.provision_sum, max_drawdown_duration,  # 7
            average_trade_duration, in_long, in_short, in_out_of_market, win_rate, annual_return, annual_std)  # 5

def backtest_wrapper(agent_type, df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic=None, reward_function=None):
    """
    # TODO add description
    """
    return generate_predictions_and_backtest(agent_type, df, agent, mkf, look_back, variables, provision, initial_balance, leverage, Trading_Environment_Basic, reward_function)


def calculate_number_of_trades_and_duration(actions):
    # Identify trade transitions
    transitions = (actions.shift(1) != actions) & (actions != 'Neutral')
    num_trades = transitions.sum()

    # Calculate durations
    durations = []
    current_duration = 0
    current_action = 'Neutral'

    # Iterate through the actions to count the consecutive non-neutral actions
    for action in actions:
        if action != 'Neutral':
            if action == current_action:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_action = action
                current_duration = 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
                current_action = 'Neutral'

    # Append the last duration if the series ended with a trade still active
    if current_duration > 0:
        durations.append(current_duration)

    # Calculate the average duration, handle cases where durations list might be empty
    avg_duration = np.mean(durations) if durations else 0

    return num_trades, avg_duration

def generate_result_statistics(df, strategy_column=None, balance_column=None, provision_sum=0, look_back=1, annualization_factor=365, starting_balance=10000):
    df = df.reset_index(drop=True)

    # Calculate returns
    returns = pd.Series([starting_balance] + df[balance_column].tolist()).pct_change().dropna()

    # Calculate final balance
    final_balance = df[balance_column].iloc[-1]
    annual_return = (final_balance / starting_balance) ** (annualization_factor / len(returns)) - 1

    # Calculate Annual Standard Deviation
    annual_std = returns.std() * np.sqrt(annualization_factor)

    # Calculate Sharpe Ratio
    sharpe_ratio = annual_return / annual_std if returns.std() > 1e-6 else float('nan')

    # Calculate Cumulative Returns
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    max_drawdown_duration = calculate_drawdown_duration(drawdown)

    # Calculate Sortino Ratio
    negative_volatility = returns[returns < 0].std() * np.sqrt(annualization_factor)
    sortino_ratio = annual_return / negative_volatility if negative_volatility > 1e-6 else float('nan')

    # Calculate Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else float('nan')

    # Calculate the number of times the agent was in long, short, or out of the market
    if strategy_column is None:
        in_long = 0
        in_short = 0
        out_of_market = 0
        num_trades = 0
        avg_duration = 0
        win_rate = 0

    else:
        # Calculate Number of Trades and Average Duration
        num_trades, avg_duration = calculate_number_of_trades_and_duration(df[strategy_column])

        # calculate profitable trades
        profitable_trades = calculate_profitable_trades(df, strategy_column, balance_column)
        win_rate = profitable_trades / num_trades if num_trades > 0 else float('nan')

        in_long = df[df[strategy_column] == 'Long'].shape[0]
        in_short = df[df[strategy_column] == 'Short'].shape[0]
        out_of_market = df[df[strategy_column] == 'Neutral'].shape[0]

    # Compile metrics
    metrics = {
        'Final Balance': df[balance_column].iloc[-1],
        'Provision Sum': provision_sum,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Max Drawdown Duration': max_drawdown_duration,
        'Calmar Ratio': calmar_ratio,
        'Number of Trades': num_trades,
        'Average trade duration': avg_duration,
        'In long': in_long / len(df),
        'In short': in_short / len(df),
        'In out of the market': out_of_market / len(df),
        'Win Rate': win_rate,
        'Annual Return': annual_return,
        'Annual Std': returns.std() * np.sqrt(annualization_factor),
    }
    return metrics


def calculate_profitable_trades(df, strategy_column, balance_column, initial_capital=10000):
    # Append a row with the initial balance at the beginning of the dataframe
    initial_row = pd.DataFrame({balance_column: [initial_capital], strategy_column: [None]})
    df = pd.concat([initial_row, df], ignore_index=True)

    # Drop consecutive duplicates in the strategy column
    filtered_df = df.loc[(df[strategy_column] != df[strategy_column].shift())]

    # Calculate profitable trades where the balance has increased from the last trade
    profitable_trades = (filtered_df[balance_column].diff() > 0).sum()

    return profitable_trades


def test_calculate_profitable_trades():
    # Test case 1: No trades
    df1 = pd.DataFrame({
        'balance_column': [10000, 10000, 10000],
        'strategy_column': ['Neutral', 'Neutral', 'Neutral']
    })
    initial_capital1 = 10000
    result1 = calculate_profitable_trades(df1, 'strategy_column', 'balance_column', initial_capital1)
    assert result1 == 0, f"Expected 0, but got {result1}"

    # Test case 3: BALANCE
    df3 = pd.DataFrame({
        'balance_column': [10000, 10100, 10100, 10300, 10400, 10500],
        'strategy_column': ['Short', 'Long', 'Neutral', 'Short', 'Long', 'Long']
    })
    initial_capital3 = 10000
    result3 = calculate_profitable_trades(df3, 'strategy_column', 'balance_column', initial_capital3)
    assert result3 == 3, f"Expected 3, but got {result3}"

    # Test case 4: BALANCE
    df4 = pd.DataFrame({
        'balance_column': [10000, 9900, 9900, 9700, 9600, 9500],
        'strategy_column': ['Neutral', 'Long', 'Neutral', 'Short', 'Neutral', 'Long']
    })
    initial_capital4 = 10000
    result4 = calculate_profitable_trades(df4, 'strategy_column', 'balance_column', initial_capital4)
    assert result4 == 0, f"Expected 0, but got {result4}"

    # Test case 5: BALANCE
    df5 = pd.DataFrame({
        'balance_column': [11000, 11000, 10050, 10150, 10000],
        'strategy_column': ['Long', 'Neutral', 'Short', 'Short', 'Long']
    })
    initial_capital5 = 10000
    result5 = calculate_profitable_trades(df5, 'strategy_column', 'balance_column', initial_capital5)
    assert result5 == 1, f"Expected 1, but got {result5}"

    print("All tests passed!")


if __name__ == '__main__':
    test_calculate_profitable_trades()

    # test average trade duration
    actions = pd.Series(['Long', 'Short'] * 5)

    num_trades, avg_duration = calculate_number_of_trades_and_duration(actions)

    print("Number of Trades:", num_trades)
    print("Average trade duration:", avg_duration)

    actions = pd.Series(['Long', 'Short', 'Neutral'] * 5)

    num_trades, avg_duration = calculate_number_of_trades_and_duration(actions)

    print("Number of Trades:", num_trades)
    print("Average trade duration:", avg_duration)

    actions = pd.Series(['Long', 'Long', 'Short', 'Neutral'] * 5)

    num_trades, avg_duration = calculate_number_of_trades_and_duration(actions)

    print("Number of Trades:", num_trades)
    print("Average trade duration:", avg_duration)
