import subprocess
import os
import torch
import pandas as pd
from time import perf_counter, sleep
from functools import wraps
from typing import Callable, Any


def get_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = perf_counter()

        print(f'"{func.__name__}()" took {end_time - start_time:.3f} seconds to execute')
        return result

    return wrapper


def generate_index_labels(rolling_datasets, dataset_type):
    index_labels = []
    for dataset in rolling_datasets:
        last_day = dataset.index[-1].strftime('%Y-%m-%d')
        label = f"{dataset_type}_{last_day}"
        index_labels.append(label)
    return index_labels

def prepare_backtest_results(backtest_results, agent_name):
    # Prepare lists to collect all data
    data = {
        'Agent Generation': [],
        'Agent Name': [],
        'Label': [],
        'Provision Sum': [],
        'Final Balance': [],
        'Total Reward': [],
        'Number of Trades': [],
        'Sharpe Ratio': [],
        'Max Drawdown': [],
        'Sortino Ratio': [],
        'Calmar Ratio': [],
        'Average Trade Duration': [],
        'Max Drawdown Duration': [],
        'In Long': [],
        'In Short': [],
        'Out of Market': []
    }

    # Extract data from backtest_results and append to lists
    for (agent_gen, label), metrics_list in backtest_results.items():
        for metrics in metrics_list:
            data['Agent Generation'].append(agent_gen)
            data['Label'].append(label)
            data['Agent Name'].append(agent_name)
            data['Provision Sum'].append(metrics.get('Provision_sum', 0))
            data['Final Balance'].append(metrics.get('Final Balance', 0))
            data['Total Reward'].append(metrics.get('Total Reward', 0))
            data['Number of Trades'].append(metrics.get('Number of Trades', 0))
            data['Sharpe Ratio'].append(metrics.get('Sharpe Ratio', 0))
            data['Max Drawdown'].append(metrics.get('Max Drawdown', 0))
            data['Sortino Ratio'].append(metrics.get('Sortino Ratio', 0))
            data['Calmar Ratio'].append(metrics.get('Calmar Ratio', 0))
            data['Average Trade Duration'].append(metrics.get('Average Trade Duration', 0))
            data['Max Drawdown Duration'].append(metrics.get('Max Drawdown Duration', 0))
            data['In Long'].append(metrics.get('In Long', 0))
            data['In Short'].append(metrics.get('In Short', 0))
            data['Out of Market'].append(metrics.get('Out of Market', 0))

    # Creating a DataFrame from the lists
    df = pd.DataFrame(data)

    # Adjusting the DataFrame to have a MultiIndex column for better organization if needed
    columns = [(df['Agent Name'].iloc[0], col) if col not in ['Label', 'Agent Generation'] else ('', col) for col in df.columns]
    multi_index = pd.MultiIndex.from_tuples(columns)
    df.columns = multi_index
    return df

def get_repository_root_path():
    try:
        # Run the git command to get the top-level directory
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT).strip().decode('utf-8')
        return repo_root
    except subprocess.CalledProcessError as e:
        print("Error getting repository root:", e.output.decode())
        return None

def save_model(self, base_dir="saved models", sub_dir="DDQN", file_name="ddqn"):
    # Get the repository root path
    repo_root = get_repository_root_path()
    if repo_root is None:
        print("Repository root not found. Using current directory as the base path.")
        repo_root = "."

    # Construct the full path
    full_path = os.path.join(repo_root, base_dir, sub_dir, file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Define the full file paths for the policy and target models
    policy_path = f"{full_path}_policy.pt"
    target_path = f"{full_path}_target.pt"

    # Save the models
    torch.save(self.q_policy.state_dict(), policy_path)
    torch.save(self.q_target.state_dict(), target_path)
    print(f"Models saved successfully: {policy_path} and {target_path}")

def save_actor_critic_model(self, base_dir="saved models", sub_dir="ActorCritic", actor_file_name="actor", critic_file_name="critic"):
    # Get the repository root path
    repo_root = get_repository_root_path()
    if repo_root is None:
        print("Repository root not found. Using current directory as the base path.")
        repo_root = "."

    # Construct the full paths
    actor_full_path = os.path.join(repo_root, base_dir, sub_dir, actor_file_name)
    critic_full_path = os.path.join(repo_root, base_dir, sub_dir, critic_file_name)

    # Ensure the directories exist
    os.makedirs(os.path.dirname(actor_full_path), exist_ok=True)
    os.makedirs(os.path.dirname(critic_full_path), exist_ok=True)

    # Define the full file paths for the actor and critic models
    actor_path = f"{actor_full_path}.pt"
    critic_path = f"{critic_full_path}.pt"

    # Save the models
    torch.save(self.actor.state_dict(), actor_path)
    torch.save(self.critic.state_dict(), critic_path)
    print(f"Models saved successfully: {actor_path} and {critic_path}")