import matplotlib.pyplot as plt
import numpy as np

def plot_total_rewards(total_rewards, model_name=None, window_size=50):
    """
    Plots the total rewards over episodes, along with a moving average of the rewards.

    Parameters:
    - total_rewards (list of int/float): List of total rewards per episode.
    - model_name (str, optional): Name of the model. Used in the plot title. Defaults to None.
    - window_size (int, optional): Size of the window for calculating the moving average. Defaults to 10.

    The function plots the original total rewards per episode and overlays a line representing the moving average of these rewards to highlight trends.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label='Total Rewards')

    # Calculate and plot moving average
    moving_avg = np.convolve(total_rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(np.arange(window_size - 1, len(total_rewards)), moving_avg, color='red', label='Moving Average')

    plt.title('Total Reward Over Episodes - ' + model_name if model_name else 'Total Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

def plot_episode_durations(episode_durations, model_name=None):
    """
    #TODO add description
    """
    plt.plot(episode_durations, color='red')
    plt.title('Episode Duration Over Episodes - ' + model_name if model_name else 'Episode Duration Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Episode Duration')
    plt.show()


def plot_total_balances(total_balances, model_name=None):
    """
    #TODO add description
    """
    plt.plot(total_balances, color='green')
    plt.title('Total Balance Over Episodes' + model_name if model_name else 'Total Balance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Balance')
    plt.show()


def plot_results(df, columns_to_plot, model_name):
    for column in columns_to_plot:
        plt.figure(figsize=(10, 6))
        for label in df['Label'].unique():
            data_subset = df[df['Label'] == label]
            plt.plot(data_subset.index, data_subset[column], marker='x', label=f"{label}")

        plt.title(f"{column} Over Generations - {model_name}")
        plt.xlabel('Agent Generation')
        plt.ylabel(column)
        plt.legend()
        plt.show()