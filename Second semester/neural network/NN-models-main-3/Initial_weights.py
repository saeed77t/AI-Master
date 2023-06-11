import numpy as np
import math


class Init_Weights:
    def __init__(self, input_size, hidden_size, output_size, bias):
        self.in_size = input_size
        self.hide_size = hidden_size
        self.out_size = output_size
        self.bias = bias

    # random initialization
    def rand_weights(self):
        self.W_h = np.random.uniform(
            low=-0.5, high=0.5, size=(self.in_size, self.hide_size)
        )
        self.W_o = np.random.uniform(
            low=-0.5, high=0.5, size=(self.hide_size, self.out_size)
        )
        if self.bias:
            bias_h = np.random.uniform(
                low=-0.5, high=0.5, size=(1, self.hide_size)
            )
            bias_o = np.random.uniform(
                low=-0.5, high=0.5, size=(1, self.out_size)
            )

            self.W_h = np.r_[bias_h, self.W_h]
            self.W_o = np.r_[bias_o, self.W_o]
        return self.W_h, self.W_o

    # have equal values
    def fill_equal(self, v_h, v_o):
        if self.bias:
            self.W_h = np.empty([self.in_size + 1, self.hide_size])
            self.W_o = np.empty([self.hide_size + 1, self.out_size])
        else:
            self.W_h = np.empty([self.in_size, self.hide_size])
            self.W_o = np.empty([self.hide_size, self.out_size])
        self.W_h.fill(v_h)
        self.W_o.fill(v_o)
        return self.W_h, self.W_o

    # Nguyen-widrow method
    def widrow(self):
        self.W_h = np.random.uniform(
            low=-0.5, high=0.5, size=(self.in_size, self.hide_size)
        )
        self.W_o = np.random.uniform(
            low=-0.5, high=0.5, size=(self.hide_size, self.out_size)
        )
        beta = 0.7 * math.pow(self.hide_size, 1 / self.in_size)

        self.W_h = np.divide(self.W_h, np.linalg.norm(self.W_h, axis=0))
        if self.bias:
            bias_h = np.random.uniform(
                low=-beta, high=beta, size=(1, self.hide_size)
            )
            bias_o = np.random.uniform(
                low=-beta, high=beta, size=(1, self.out_size)
            )

            self.W_h = np.r_[bias_h, self.W_h]
            self.W_o = np.r_[bias_o, self.W_o]
        return self.W_h, self.W_o

    # Le-cunn intialization weights with variance (1/n, 1/p)
    # and zero mean and uniform disturbution
    def LeCunn(self):
        b = 0.5 * np.sqrt(12 / self.in_size)
        a = -b
        self.W_h = np.random.uniform(
            low=a, high=b, size=(self.in_size, self.hide_size)
        )
        self.W_o = np.random.uniform(
            low=a, high=b, size=(self.hide_size, self.out_size)
        )

        if self.bias:
            bias_h = np.random.uniform(low=a, high=b, size=(1, self.hide_size))
            bias_o = np.random.uniform(low=a, high=b, size=(1, self.out_size))

            self.W_h = np.r_[bias_h, self.W_h]
            self.W_o = np.r_[bias_o, self.W_o]
        return self.W_h, self.W_o
