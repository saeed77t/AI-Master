import numpy as np
from delta_rule import delta_rule
from activation_func import activation_func


class adaline:
    def __init__(
        self, sample_size: int, f, d_f, w=None, b=0, a=1, printing=False
    ) -> None:
        self.sample_size = sample_size
        self.f = f
        self.d_f = d_f
        if w is None:
            self.w = np.zeros((sample_size, sample_size))
        else:
            self.w = w
        self.table = np.empty((0, sample_size))
        self.printing = printing

    def train(self, S, t):
        self.model = delta_rule(
            input_size=self.sample_size,
            f=activation_func.identity,
            d_f=activation_func.d_identity,
            printing=self.printing,
        )
        self.model.train(S, t)

    def print_table(self):
        self.model.print_table()


def test_adaline():
    data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    model = adaline(
        sample_size=data.shape[1],
        f=activation_func.identity,
        d_f=activation_func.d_identity,
        printing=True,
    )

    labels = np.array([1, -1, -1, -1])

    model.train(data, labels)
    model.print_table()


def example2():
    data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    labels = np.array([0, 0, 0, 1, 1])

    model = adaline(
        sample_size=data.shape[1],
        f=activation_func.identity,
        d_f=activation_func.d_identity,
        printing=True,
    )

    model.train(data, labels)
    model.print_table()

example2()
# test_adaline()
