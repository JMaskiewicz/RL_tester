import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Process, Queue, Event, Manager
from threading import Thread
import sys
import time
from time import perf_counter, sleep
from functools import wraps
from typing import Callable, Any
from numba import jit
import math


import backtest.backtest_functions.functions as BF
from trading_environment.environment import Trading_Environment_Basic

"""
Description of parallelization of the environment
Initiate the learning phase as soon as enough experiences are collected, then immediately resume experience gathering after learning while backtesting is performed in parallel for the updated agent. 
This approach ensures continuous data collection and efficient utilization of computational resources.
"""

def get_time(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = perf_counter()

        print(f'"{func.__name__}()" took {end_time - start_time:.3f} seconds to execute')
        return result

    return wrapper

def print_signal_status(arg1=None, **kwargs):
    def print_status(signals):
        print(f"Status of Signals:")
        for key, value in signals.items():
            if isinstance(value, list):
                for i, signal in enumerate(value, start=1):
                    status = 'Set' if signal.is_set() else 'Not Set'
                    print(f"{key} {i}: {status}")
            else:
                status = 'Set' if value.is_set() else 'Not Set'
                print(f"{key}: {status}")

    if callable(arg1):
        func = arg1
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self = args[0]

            print(f"Status of Signals after {func.__name__}:")
            signals = {
                'Work Event': self.work_event,
                'Pause Signals': self.pause_signals,
                'Resume Signals': self.resume_signals,
                'Backtesting Completed': self.backtesting_completed,
            }
            print_status(signals)
            return result

        return wrapper
    else:
        print_status(arg1 or kwargs)

def progress_update(interval=60):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Dynamically retrieve arguments based on function signature
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            arg_dict = dict(zip(arg_names, args))
            arg_dict.update(kwargs)

            # Ensure required arguments are present
            required_args = ['shared_episodes_counter', 'max_episodes_per_worker', 'num_workers']
            missing_args = [arg for arg in required_args if arg not in arg_dict]
            if missing_args:
                raise ValueError(f"Missing required argument(s): {', '.join(missing_args)}")

            def progress_updater(stop_event, shared_episodes_counter, max_episodes_per_worker, num_workers):
                """Inner function to update the progress based on shared counter and total episodes."""
                start_time = time.time()
                while not stop_event.is_set():
                    current_episodes = shared_episodes_counter.value
                    total_episodes = max_episodes_per_worker * num_workers
                    progress_percentage = (current_episodes / total_episodes) * 100
                    bar_length = 50
                    filled_length = int(round(bar_length * progress_percentage / 100))
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    elapsed_time = time.time() - start_time
                    speed = current_episodes / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = ((total_episodes - current_episodes) / speed) if speed > 0 else 0
                    formatted_elapsed_time = format_time(elapsed_time)
                    formatted_eta = format_time(eta_seconds)
                    sys.stdout.write(
                        f"\033[92m\rProgress: {progress_percentage:.0f}% |{bar}| {current_episodes}/{total_episodes} [Elapsed: {formatted_elapsed_time}, ETA: {formatted_eta}, Speed: {speed:.2f}it/s]\033[0m")
                    sys.stdout.flush()
                    time.sleep(interval)

            # Start the progress updater thread
            stop_event = Event()
            updater_thread = Thread(target=progress_updater, args=(
                stop_event,
                arg_dict['shared_episodes_counter'],
                arg_dict['max_episodes_per_worker'],
                arg_dict['num_workers']
            ))
            updater_thread.start()

            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                updater_thread.join()
            return result

        return wrapper
    return decorator


@get_time
#@print_signal_status
def backtest_in_background(agent_type, agent, backtest_results, num_workers_backtesting, val_rolling_datasets, test_rolling_datasets, val_labels, test_labels, probs_dfs, balances_dfs, backtesting_completed, reward_function):
    start_time = time.time()
    print("Starting backtesting...")
    with ThreadPoolExecutor(max_workers=num_workers_backtesting) as executor:
        futures = []
        for df, label in zip(val_rolling_datasets + test_rolling_datasets, val_labels + test_labels):
            future = executor.submit(BF.backtest_wrapper, agent_type, df, agent, 'EURUSD', look_back, variables, provision, starting_balance, leverage, Trading_Environment_Basic, reward_function)
            futures.append((future, label))

        for future, label in futures:
            (balance, total_reward, number_of_trades, probs_df, action_df, buy_and_hold_return, sell_and_hold_return, sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio, cumulative_returns, balances) = future.result()
            result_data = {
                'Agent generation': agent.generation,
                'Label': label,
                'Final Balance': balance,
                'Total Reward': total_reward,
                'Number of Trades': number_of_trades,
                'Buy and Hold Return': buy_and_hold_return,
                'Sell and Hold Return': sell_and_hold_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio
            }
            key = (agent.generation, label)
            if key not in backtest_results:
                backtest_results[key] = []
            backtest_results[key].append(result_data)
            # Store probabilities and balances for plotting
            probs_dfs[(agent.generation, label)] = probs_df
            balances_dfs[(agent.generation, label)] = balances

    # Signal that backtesting is completed
    backtesting_completed.set()
    end_time = time.time()
    episode_time = end_time - start_time
    print(f"Backtesting completed in {episode_time:.2f} seconds\n")


def environment_worker(agent_type, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signal, resume_signal, total_rewards, total_balances, worker_id, individual_worker_batch_size, workers_completed, workers_completed_signal, shared_episodes_counter):
    random.seed(worker_id)  # Seed the random number generator with a unique seed for this worker

    experiences_collected = 0  # Initialize a counter for collected experiences
    start_time = time.time()  # Record the start time of data collection

    for episode in range(max_episodes_per_worker):
        shared_episodes_counter.value += 1
        df = random.choice(dfs)
        env = Trading_Environment_Basic(df, **env_settings)
        observation = env.reset()
        done = False

        while not done:
            work_event.wait()  # Wait for permission to work
            if resume_signal.is_set():
                print(f"Worker {multiprocessing.current_process().name}: Starting...")
                experiences_collected = 0  # Reset the counter after pausing
                start_time = time.time()  # Reset the start time for the next batch
                resume_signal.clear()  # Clear the resume signal

            action, prob, val = agent.choose_action(observation, env.current_position)
            observation_, reward, done, info = env.step(action)
            experience = (observation, action, prob, val, reward, done, env.current_position)
            shared_queue.put(experience)
            observation = observation_

            experiences_collected += 1  # Increment the counter for each experience collected
            if experiences_collected >= individual_worker_batch_size:
                end_time = time.time()  # Record the time when the batch size limit is reached
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                print(f"Worker {multiprocessing.current_process().name}: Reached individual batch size limit in {elapsed_time:.2f} seconds.")
                pause_signal.set()
                # print_signal_status({'Pause Signal': pause_signal, 'backtesting_completed': workers_completed_signal, 'Work Event': work_event, 'Resume Signal': resume_signal})
                while pause_signal.is_set():
                    time.sleep(0.1)

            if workers_completed_signal.is_set():
                break

        # Append the total reward and final balance for this episode to the shared lists
        total_rewards.append(env.reward_sum)
        total_balances.append(env.balance)
        print(f"Worker {multiprocessing.current_process().name} completed training df of length {len(df)}, first observation in training df is {df.index[0]}, episode {episode+1}/{max_episodes_per_worker} with cumulative reward {env.reward_sum} and final balance {env.balance}")

    print(f"Worker {multiprocessing.current_process().name} has completed all tasks.")
    workers_completed.value += 1
    if workers_completed.value >= 1:
        workers_completed_signal.set()

# TODO add description
# TODO add early stopping based on the validation set from the backtesting

@get_time
def collect_and_learn(agent_type, dfs, max_episodes_per_worker, env_settings, batch_size_for_learning, backtest_results, agent,
                      num_workers, num_workers_backtesting, backtesting_frequency=1):

    manager = Manager()
    total_rewards, total_balances, shared_episodes_counter, workers_completed, backtesting_completed, work_event, pause_signals, resume_signals, workers_completed_signal, shared_queue = setup_shared_resources_and_events(
        manager, num_workers)

    workers = start_workers(agent_type, num_workers, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event,
                            pause_signals, resume_signals, total_rewards, total_balances, workers_completed,
                            workers_completed_signal, shared_episodes_counter, batch_size_for_learning)

    manage_learning_and_backtesting(agent_type, agent, num_workers_backtesting, backtest_results, backtesting_completed, work_event,
                                    pause_signals, resume_signals, shared_queue, workers_completed_signal,
                                    shared_episodes_counter, total_rewards, total_balances, batch_size_for_learning,
                                    backtesting_frequency, max_episodes_per_worker, num_workers)

    for worker in workers:
        worker.join()

    print("\r" + " " * 100, end='')
    print("All workers stopped.")
    return list(total_rewards), list(total_balances)

@get_time
@progress_update(interval=10)
def manage_learning_and_backtesting(agent_type, agent, num_workers_backtesting, backtest_results, backtesting_completed, work_event, pause_signals, resume_signals, shared_queue, workers_completed_signal, shared_episodes_counter, total_rewards, total_balances, batch_size_for_learning, backtesting_frequency, max_episodes_per_worker=10, num_workers=4):
    agent_generation = 0
    try:
        total_experiences = 0
        while True:
            if not work_event.is_set():
                work_event.set()

            while not shared_queue.empty():
                experience = shared_queue.get_nowait()
                agent.store_transition(*experience)
                total_experiences += 1
                # print('EXP ', total_experiences)

            if total_experiences >= batch_size_for_learning and backtesting_completed.is_set():
                print("\nLearning phase initiated.")
                agent.learn()
                total_experiences = 0
                agent.memory.clear_memory()

                for signal in pause_signals:
                    signal.clear()
                work_event.set()
                for resume_signal in resume_signals:
                    resume_signal.set()

                if agent.generation > agent_generation and agent.generation % backtesting_frequency == 0:
                    backtesting_completed.clear()
                    backtest_thread = Thread(target=backtest_in_background, args=(agent_type, agent, backtest_results, num_workers_backtesting, val_rolling_datasets, test_rolling_datasets, val_labels, test_labels, probs_dfs, balances_dfs, backtesting_completed))
                    backtest_thread.start()
                    agent_generation = agent.generation

            if workers_completed_signal.is_set():
                backtesting_completed.wait()
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    print("All workers stopped.")
    return None

def setup_shared_resources_and_events(manager, num_workers):
    total_rewards = manager.list()
    total_balances = manager.list()
    shared_episodes_counter = manager.Value('i', 0)
    workers_completed = manager.Value('i', 0)
    backtesting_completed = Event()
    backtesting_completed.set()
    work_event = Event()
    pause_signals = [Event() for _ in range(num_workers)]
    resume_signals = [Event() for _ in range(num_workers)]
    workers_completed_signal = Event()
    shared_queue = manager.Queue()
    return total_rewards, total_balances, shared_episodes_counter, workers_completed, backtesting_completed, work_event, pause_signals, resume_signals, workers_completed_signal, shared_queue

def start_workers(agent_type, num_workers, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signals, resume_signals, total_rewards, total_balances, workers_completed, workers_completed_signal, shared_episodes_counter, batch_size_for_learning):
    workers = []
    for i in range(num_workers):
        worker_process = Process(target=environment_worker, args=(agent_type, dfs, shared_queue, max_episodes_per_worker, env_settings, agent, work_event, pause_signals[i], resume_signals[i], total_rewards, total_balances, i+1, batch_size_for_learning // num_workers, workers_completed, workers_completed_signal, shared_episodes_counter))
        worker_process.start()
        workers.append(worker_process)
    return workers

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h:{m}m:{s}s"