from numba import jit
import math

@jit(nopython=True)
def reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision):
    # Calculate the log return
    if previous_close != 0 and current_close != 0:
        log_return = math.log(current_close / previous_close)
    else:
        log_return = 0

    # Calculate the base reward
    reward = log_return * current_position * leverage

    # Penalize the agent for taking the wrong action
    if reward < 0:
        reward *= 2.5  # penalty for wrong action

    # Calculate the cost of provision if the position has changed, and it's not neutral (0).
    if current_position != previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 - provision) * 10  # penalty for changing position
    elif current_position == previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 + provision) * 1  # small premium for holding position
    else:
        provision_cost = 0

    # Apply the provision cost
    reward += provision_cost

    # Scale the reward to enhance its significance for the learning process
    final_reward = reward * 100

    return final_reward

def reward_calculation_nojit(previous_close, current_close, previous_position, current_position, leverage, provision):
    # Calculate the log return
    if previous_close != 0 and current_close != 0:
        log_return = math.log(current_close / previous_close)
    else:
        log_return = 0

    # Calculate the base reward
    reward = log_return * current_position * leverage

    # Penalize the agent for taking the wrong action
    if reward < 0:
        reward *= 2.5  # penalty for wrong action

    # Calculate the cost of provision if the position has changed, and it's not neutral (0).
    if current_position != previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 - provision) * 10  # penalty for changing position
    elif current_position == previous_position and abs(current_position) == 1:
        provision_cost = math.log(1 + provision) * 1  # small premium for holding position
    else:
        provision_cost = 0

    # Apply the provision cost
    reward += provision_cost

    # Scale the reward to enhance its significance for the learning process
    final_reward = reward * 100

    return final_reward

import time

# Example parameters for the function
previous_close = 100
current_close = 105
previous_position = 1
current_position = -1
leverage = 10
provision = 0.001

# Benchmark the JIT version
start_time = time.time()
for _ in range(50000000):
    reward_calculation(previous_close, current_close, previous_position, current_position, leverage, provision)
end_time = time.time()
jit_duration = end_time - start_time

# Benchmark the non-JIT version
start_time = time.time()
for _ in range(50000000):
    reward_calculation_nojit(previous_close, current_close, previous_position, current_position, leverage, provision)
end_time = time.time()
nojit_duration = end_time - start_time

print(f"JIT version took: {jit_duration} seconds")
print(f"Non-JIT version took: {nojit_duration} seconds")