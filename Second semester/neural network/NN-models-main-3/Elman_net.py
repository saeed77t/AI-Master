import numpy as np
from activation_func import activation_func


class ElmanNet:
    def __init__(self, f_hidden, f_output) -> None:
        self.f_hidden = f_hidden
        self.f_output = f_output

    def feed(self, x, h_0, W_IH, W_HH, W_HO):
        h_in = np.dot(x, W_IH) + np.dot(h_0, W_HH)
        h = self.f_hidden(h_in)
        y_in = np.dot(h, W_HO)
        y = self.f_output(y_in)
        return h, y


f_h = activation_func.bipolar_sigmoid
f_o = activation_func.identity

x = np.array([[1, -1, 1]])
h = np.array([[-1, 1, -1, 1]])
W_IH = np.array([
    [0.5, -0.3, 0.3, 0.2],
    [0.3, 0.4, -0.4, 0.3],
    [-0.2, 0.3, 0.3, 0.4]
])
W_HH = np.array([
    [0, 1, -1, 0],
    [-1, 0, -1, 1],
    [1, 1, 0, -1],
    [0, -1, 1, 0]

])
W_HO = np.array([
    [0.1, -0.2],
    [-0.3, 0.3],
    [-0.3, 0.1],
    [0.2, -0.2]
])

Elman = ElmanNet(f_hidden=f_h, f_output=f_o)
net = Elman.feed(x=x, h_0=h, W_IH=W_IH, W_HH=W_HH, W_HO=W_HO)
print(net)
