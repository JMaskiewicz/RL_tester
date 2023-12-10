import numpy as np

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def standardize_data(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val
