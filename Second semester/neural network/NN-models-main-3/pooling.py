import numpy as np


class pooling:
    def average_pooling(x, pooling_size, stride_pooling, params = None):
        pooling_output = []
        for j in range(0, len(x) - pooling_size + 1, stride_pooling):
            pooling_output.append(np.mean(x[j : j + pooling_size]))
        return pooling_output
    
    def weighted_pooling(x, pooling_size, stride_pooling, params = None):
        pooling_output = []
        for j in range(0, len(x) - pooling_size + 1, stride_pooling):
            pooling_output.append(np.average(x[j : j + pooling_size], weights=params))
        return pooling_output

    def max_pooling(x, pooling_size, stride_pooling, params = None):
        pooling_output = []
        for j in range(0, len(x) - pooling_size + 1, stride_pooling):
            pooling_output.append(np.max(x[j : j + pooling_size]))
        return pooling_output

    def l2_norm_pooling(x, pooling_size, stride_pooling, params = None):
        pooling_output = []
        for j in range(0, len(x) - pooling_size + 1, stride_pooling):
            pooling_output.append(np.linalg.norm(x[j : j + pooling_size]))
        return pooling_output
