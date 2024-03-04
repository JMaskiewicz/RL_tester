import matplotlib.pyplot as plt

def plot_total_rewards(total_rewards, model_name=None):
    """
    #TODO add description
    """
    plt.plot(total_rewards)
    plt.title('Total Reward Over Episodes - ' + model_name if model_name else 'Total Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
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