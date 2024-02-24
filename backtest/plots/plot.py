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

def plot_pnl_over_episodes(num_episodes, validation_pnl, test_pnl, model_name=None):
    """
    #TODO add description
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episodes), validation_pnl.values, label='Validation Total PnL', marker='o')
    plt.plot(range(num_episodes), test_pnl.values, label='Test Total PnL', marker='x')
    plt.title('Total PnL Over Episodes' + model_name if model_name else 'Total PnL Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total PnL')
    plt.legend()
    plt.show()

def plot_action_probabilities(probabilities, figsize=(16, 6), data_set=None, model_name=None):
    """
    #TODO add description
    """
    plt.figure(figsize=figsize)
    plt.plot(probabilities['Short'], label='Short', color='red')
    plt.plot(probabilities['Neutral'], label='Neutral', color='blue')
    plt.plot(probabilities['Long'], label='Long', color='green')
    plt.title(f'Action Probabilities Over Time {data_set} - {model_name}' if model_name else 'Action Probabilities Over Time')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


def plot_all(total_rewards=None, episode_durations=None, total_balances=None, num_episodes=None, validation_pnl=None,
             test_pnl=None, probabilities_sets=None, plot_rewards=True, plot_durations=True, plot_balances=True,
             plot_pnl=True, plot_probabilities=True, model_name=None):
    """
    Plot all the metrics
    #TODO add description for each parameter
    """

    if plot_rewards and total_rewards is not None:
        plot_total_rewards(total_rewards, model_name)

    if plot_durations and episode_durations is not None:
        plot_episode_durations(episode_durations, model_name)

    if plot_balances and total_balances is not None:
        plot_total_balances(total_balances, model_name)

    if plot_pnl and num_episodes is not None and validation_pnl is not None and test_pnl is not None:
        plot_pnl_over_episodes(num_episodes, validation_pnl, test_pnl, model_name)

    if plot_probabilities and probabilities_sets:
        for data_set, probabilities in probabilities_sets.items():
            plot_action_probabilities(probabilities, data_set=data_set, model_name=model_name)