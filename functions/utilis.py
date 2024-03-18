import subprocess
import os
import torch


def generate_index_labels(rolling_datasets, dataset_type):
    index_labels = []
    for dataset in rolling_datasets:
        last_day = dataset.index[-1].strftime('%Y-%m-%d')
        label = f"{dataset_type}_{last_day}"
        index_labels.append(label)
    return index_labels

def prepare_backtest_results(backtest_results):
    agent_generations = []
    labels = []
    final_balances = []
    total_rewards = []
    number_of_trades = []
    buy_and_hold_returns = []
    sell_and_hold_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    sortino_ratios = []
    calmar_ratios = []

    # Iterating over each record in the dataset
    for (agent_gen, label), metrics in backtest_results.items():
        for metric in metrics:
            agent_generations.append(agent_gen)
            labels.append(label)
            final_balances.append(metric['Final Balance'])
            total_rewards.append(metric['Total Reward'])
            number_of_trades.append(metric['Number of Trades'])
            buy_and_hold_returns.append(metric['Buy and Hold Return'])
            sell_and_hold_returns.append(metric['Sell and Hold Return'])
            sharpe_ratios.append(metric['Sharpe Ratio'])
            max_drawdowns.append(metric['Max Drawdown'])
            sortino_ratios.append(metric['Sortino Ratio'])
            calmar_ratios.append(metric['Calmar Ratio'])

    # Creating a DataFrame from the lists
    df = pd.DataFrame({
        'Agent Generation': agent_generations,
        'Label': labels,
        'Final Balance': final_balances,
        'Total Reward': total_rewards,
        'Number of Trades': number_of_trades,
        'Buy and Hold Return': buy_and_hold_returns,
        'Sell and Hold Return': sell_and_hold_returns,
        'Sharpe Ratio': sharpe_ratios,
        'Max Drawdown': max_drawdowns,
        'Sortino Ratio': sortino_ratios,
        'Calmar Ratio': calmar_ratios,
    })
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