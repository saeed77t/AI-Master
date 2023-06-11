import numpy as np
from tabulate import tabulate
from activation_func import activation_func
from hebb_rule import hebb_rule
from delta_rule import delta_rule


class AAM:
    def __init__(self, sample_size: int, rule, f, printing=False) -> None:
        self.sample_size = sample_size
        self.rule = rule
        self.f = f
        self.w = np.zeros((sample_size, sample_size))
        self.table = np.empty((0, sample_size))
        self.printing = printing

    def train_sample(self, S: np.ndarray):
        """
        Get one sample and update the weight matrix
        """
        w = []
        for i in range(self.sample_size):
            delta_w, _ = self.rule.train_sample(S, S[i])
            self.rule.reset_weights()
            w.append(delta_w)

        w = np.array(w).T - np.eye(self.sample_size)

        if self.printing:
            self.table = np.vstack((self.table, w))

        return w

    def train(self, S: np.ndarray):
        """
        Get a set of samples and update the weight matrix
        """

        for i in range(S.shape[0]):
            self.w += self.train_sample(S[i])

        if self.printing:
            self.table = np.vstack((self.table, self.w))

    def print_table(self):
        n_sample = int(len(self.table) / self.sample_size) - 1
        for i in range(n_sample):
            print(f"w({i+1}):")
            print(
                tabulate(
                    self.table[
                        i * self.sample_size : (i + 1) * self.sample_size
                    ],
                    tablefmt="fancy_grid",
                )
            )
        print(f"W':")
        print(
            tabulate(
                self.table[n_sample * self.sample_size :],
                tablefmt="fancy_grid",
            )
        )

    def recall(self, X: np.ndarray):
        """
        Get a sample and return the output
        """
        y_in = self.w.T @ X
        if self.printing:
            print("X:", X)
            print("y_in:", y_in)
        y = np.array([self.f(y_in[i]) for i in range(self.sample_size)])
        return y


def test_AAM_n_1():
    S = np.array([[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1]])

    heb = hebb_rule(4)
    model = AAM(4, heb, activation_func.sign, True)

    model.train(S, True)
    model.print_table()

    sample = np.array([1, -1, 1, 1])
    print(model.recall(sample, True))


def test_AAM_n():
    S = np.array([[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])

    heb = hebb_rule(4)
    model = AAM(4, heb, activation_func.sign, True)

    model.train(S, True)
    model.print_table()

    sample = np.array([1, 1, 1, -1])
    print(model.recall(sample, True))


def test_pre_mid():
    s1 = [-1, -1, 1, 1]
    s2 = [-1, 1, -1, 1]
    s3 = [1, -1, 1, -1]
    s4 = [1, 1, -1, -1]
    S = np.array([s1, s2, s3, s4])

    heb = hebb_rule(4)
    model = AAM(sample_size=4, rule=heb, f=activation_func.sign, printing=True)

    model.train(S)

    model.print_table()

    print(model.recall(s1))
    print(model.recall(s2))
    print(model.recall(s3))
    print(model.recall(s4))


test_pre_mid()
