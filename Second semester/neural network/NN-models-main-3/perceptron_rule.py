from copy import copy
import numpy as np
from tabulate import tabulate
from activation_func import activation_func


class perceptron_rule:
    def __init__(self, input_size: int, f, w=None, b=0, a=1, printing=False):
        self.input_size = input_size
        self.f = f
        if w is None:
            self.w = np.zeros(input_size)
        else:
            self.w = w
        self.b = b
        self.a = a  # Learning rate
        self.table = np.empty((0, 2 * input_size + 8))
        self.printing = printing

    def train_sample(self, X: np.ndarray, t: int):
        """
        Get one sample and update the weight matrix
        """
        y_in = self.b + X @ self.w
        y = self.f(y_in)

        if self.printing:
            row = np.append(X, 1)
            row = np.append(row, y_in)
            row = np.append(row, y)
            row = np.append(row, t)

        if y != t:
            delta_w = self.a * X * t
            delta_b = self.a * t
        else:
            delta_w = np.zeros(self.input_size)
            delta_b = 0

        self.w += delta_w
        self.b += delta_b

        if self.printing:
            row = np.append(row, delta_w)
            row = np.append(row, delta_b)
            row = np.append(row, self.w)
            row = np.append(row, self.b)

            self.table = np.vstack((self.table, row))

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Get a set of samples and update the weight matrix
        """
        w = np.random.rand(self.input_size)
        epoch = 1
        while not np.array_equal(w, self.w):
            if self.printing:
                self.table = np.vstack(
                    (
                        self.table,
                        np.full(2 * self.input_size + 8, f"epc {epoch}"),
                    )
                )
                epoch += 1
            w = copy(self.w)
            for i in range(len(X)):
                self.train_sample(X[i], y[i])

    def print_table(self):
        # Set columns labels
        labels = ["x" + str(i) for i in range(self.input_size)]
        labels.append("1")
        labels.append("y_in")
        labels.append("y")
        labels.append("t")
        for i in range(self.input_size):
            labels.append("Δw" + str(i))
        labels.append("Δb")
        for i in range(self.input_size):
            labels.append("w" + str(i))
        labels.append("b")

        print(tabulate(self.table, headers=labels, tablefmt="fancy_grid"))


def test_perceptron_rule():
    model = perceptron_rule(input_size=2, f=activation_func.sign, a=1, printing=True)

    data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    labels = np.array([-1, 1, 1, 1])

    model.train(data, labels)
    model.print_table()


test_perceptron_rule()
