import numpy as np


def normalize(x):
    x_min = min(x)
    x_max = max(x)
    m = len(x)
    x_norm = np.array([])

    for i in range(m):
        x_norm = np.append(x_norm, (x[i] - x_min) / (x_max - x_min))

    # we reshape the data for the next matrix multiplications
    x_norm = x_norm.reshape(m, 1)
    return x_norm


def add_bias(x):
    return np.hstack([np.ones([x.shape[0], 1]), x])
