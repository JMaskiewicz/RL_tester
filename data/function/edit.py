from numba import jit
import numpy as np

# this speed up calculations by 10% (3s per episode)
@jit(nopython=True)
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

@jit(nopython=True)
def standardize_data(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    standardized = (data - mean_val) / std_val
    return standardized

def process_variable(data, edit_type):
    if edit_type == 'standardize':
        return standardize_data(data)
    elif edit_type == 'normalize':
        return normalize_data(data)
    else:
        return data