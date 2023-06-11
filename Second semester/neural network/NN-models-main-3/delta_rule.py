from copy import copy
import numpy as np
from tabulate import tabulate
from activation_func import activation_func


class delta_rule:
    def __init__(
        self, input_size: int, f, d_f, w=None, b=0, a=0.2, printing=False
    ):
        self.input_size = input_size
        self.f = f
        self.d_f = d_f
        if w is None:
            self.w = np.zeros(input_size)
        else:
            self.w = w
        self.b = b
        self.a = a  # Learning rate
        self.table = np.empty((0, 3 * input_size + 6))
        self.tolerance = 0.2
        self.printing = printing

    def reset_weights(self):
        self.w = np.zeros(self.input_size)
        self.b = 0

    def train_sample(self, X: np.ndarray, t: int):
        """
        Get one sample and update the weight matrix
        """
        y_in = self.b + X @ self.w
        y = self.f(y_in)

        if self.printing:
            row = np.append(X, 1)
            row = np.append(row, t)
            row = np.append(row, y_in)
            row = np.append(row, y)

        delta_w = self.a * (t - y) * self.d_f(y_in) * X
        delta_b = self.a * (t - y) * self.d_f(y_in)

        self.w += delta_w
        self.b += delta_b

        if self.printing:
            row = np.append(row, delta_w)
            row = np.append(row, delta_b)
            row = np.append(row, self.w)
            row = np.append(row, self.b)

            self.table = np.vstack((self.table, np.round(row, 4)))

        return delta_w, delta_b

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Get a set of samples and update the weight matrix
        """
        w = np.random.rand(self.input_size)
        epoch = 1
        # While delta w is greater than tolerance
        while np.sum(np.abs(w - self.w)) > self.tolerance:
            if self.printing:
                self.table = np.vstack(
                    (
                        self.table,
                        np.full(3 * self.input_size + 6, epoch),
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
        labels.append("t")
        labels.append("y_in")
        labels.append("y")
        for i in range(self.input_size):
            labels.append("Δw" + str(i))
        labels.append("Δb")
        for i in range(self.input_size):
            labels.append("w" + str(i))
        labels.append("b")

        print(tabulate(self.table, headers=labels, tablefmt="fancy_grid"))


def test_delta_rule():
    init_w = np.array([0.1, 0.3])
    model = delta_rule(
        input_size=2,
        f=activation_func.bipolar_sigmoid,
        d_f=activation_func.d_bipolar_sigmoid,
        w=init_w,
        b=0.2,
        a=0.2,
        printing=True,
    )

    data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    labels = np.array([1, -1, -1, -1])

    model.train(data, labels)
    model.print_table()


def midterm_p7():
    # NAND logical function
    data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    labels = np.array([-1, 1, 1, 1])

    init_w = np.array([1.0, 1.0])
    model = delta_rule(
        input_size=data.shape[1],
        f=activation_func.bipolar_sigmoid,
        d_f=activation_func.d_bipolar_sigmoid,
        w=init_w,
        b=1.0,
        a=0.5,
        printing=True,
    )

    model.train(data, labels)
    model.print_table()


# midterm_p7()

# test_delta_rule()
