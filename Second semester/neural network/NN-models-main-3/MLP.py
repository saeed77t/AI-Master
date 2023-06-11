import numpy as np
from activation_func import activation_func
from Initial_weights import Init_Weights
from tabulate import tabulate


class MLP:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_func_hidden,
        activation_func_output,
        derivative_act_hidden,
        derivative_act_output,
        bias=True,
        batch_mode=True,
        print_steps=True,
        decimal_point=2,
        init_weights=None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias
        self.batch = batch_mode
        self.dp = decimal_point
        self.print_steps = print_steps
        if(init_weights == None):
            init_weights = Init_Weights(
                self.input_size, self.hidden_size, self.output_size, bias=self.bias
            )
            self.W1, self.W2 = init_weights.fill_equal(v_h=0.5, v_o=0.5)
        else:
            self.W1, self.W2 = init_weights

        self.delta_W1 = np.zeros_like(self.W1)
        self.delta_W2 = np.zeros_like(self.W2)
        self.act_func_h = activation_func_hidden
        self.act_func_o = activation_func_output
        self.de_act_h = derivative_act_hidden
        self.de_act_o = derivative_act_output
        self.o = 0
        self.h = 0

    def forward(self, X):
        """
        X: input sample
        """
        if self.bias:
            X = np.c_[np.ones(X.shape[0]), X]
        self.z_in = np.dot(X, self.W1)
        self.z = self.act_func_h(self.z_in)
        if self.bias:
            self.z = np.c_[np.ones(self.z.shape[0]), self.z]
        self.y_in = np.dot(self.z, self.W2)
        output = self.act_func_o(self.y_in)
        return output

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = -1 * self.output_error * self.de_act_o(self.y_in)
        self.sigma_output_delta = self.output_delta.dot(self.W2[1:, :].T)
        self.hidden_delta = self.sigma_output_delta * self.de_act_h(self.z_in)
        if self.batch:
            self.h_q = X.T.dot(self.hidden_delta)
            self.h += self.h_q
            self.o_q = self.z.T.dot(self.output_delta)
            self.o += self.o_q

    def train(self, X, y, epochs, learning_rate):
        if self.batch:
            for epoch in range(epochs):
                print(f'epoch {epoch+1}')
                table_1 = []
                table_2 = []
                for i in range(X.shape[0]):
                    inp = X[i].reshape([1, -1])
                    output = self.forward(inp)
                    self.backward(inp, y[i], output)
                    table_1.append([inp, np.round(self.z_in, self.dp), np.round(self.z, self.dp),
                                    np.round(self.y_in, self.dp), np.round(self.y_in, self.dp)])
                    table_2.append([np.round(self.hidden_delta, self.dp), np.round(self.output_delta, self.dp),
                                    np.round(self.h_q, self.dp), np.round(self.o_q, self.dp)])

                self.delta_v = (self.h / X.shape[0]) * learning_rate * -1
                self.delta_w = (self.o / X.shape[0]) * learning_rate * -1
                self.W1 += self.delta_v
                self.W2 += self.delta_w

                if(self.print_steps):
                    print(tabulate(table_1, headers=[
                        "x", "z_in", "z", "y_in", "y"]))
                    print(tabulate(table_2, headers=[
                        "δ_H", "δ_O", "h(q)", "o(q)"]))

                    print(f"o = {np.round(self.o, self.dp)}")
                    print(f"h = {np.round(self.h, self.dp)}")
                    print(f"Δv = {np.round(self.delta_v, self.dp)}")
                    print(f"Δw = {np.round(self.delta_w, self.dp)}")
                    print(f"v = {np.round(self.W1, self.dp)}")
                    print(f"w = {np.round(self.W2, self.dp)}")
        else:
            for epoch in range(epochs):
                print(f'epoch {epoch+1}')
                table_1 = []
                table_2 = []
                for i in range(X.shape[0]):
                    inp = X[i].reshape([1, -1])
                    output = self.forward(inp)
                    self.backward(inp, y[i], output)
                    self.delta_v = (
                        inp.T.dot(self.hidden_delta) * learning_rate * -1
                    )
                    self.delta_w = (
                        self.z.T.dot(self.output_delta) * learning_rate * -1
                    )
                    self.W1 += self.delta_v
                    self.W2 += self.delta_w
                    table_1.append([inp, np.round(self.z_in, self.dp), np.round(self.z, self.dp), np.round(self.y_in, self.dp),
                                   np.round(self.y_in, self.dp)])
                    table_2.append([np.round(self.hidden_delta, self.dp), np.round(self.output_delta, self.dp), np.round(
                        self.delta_v, self.dp), np.round(self.delta_w, self.dp), np.round(self.W1, self.dp), np.round(self.W2, self.dp)])

                if(self.print_steps):
                    print(tabulate(table_1, headers=[
                        "x", "z_in", "z", "y_in", "y"]))
                    print(tabulate(table_2, headers=[
                        "δ_H", "δ_O", "Δv", "Δw", "v", "w"]))
        return self.W1, self.W2


def example():
    # Example usage
    X = np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.25]])
    y = np.array([[0.25], [0.75]])
    input_size = X.shape[1]
    hidden_size = 2
    output_size = y.shape[1]

    hidden_act = activation_func.bipolar_sigmoid
    de_hidden_act = activation_func.d_bipolar_sigmoid
    output_act = activation_func.bipolar_sigmoid
    de_output_act = activation_func.d_bipolar_sigmoid
    mlp = MLP(
        input_size,
        hidden_size,
        output_size,
        activation_func_hidden=hidden_act,
        activation_func_output=output_act,
        derivative_act_hidden=de_hidden_act,
        derivative_act_output=de_output_act,
        batch_mode=False,
        bias=False,
        print_steps=True
    )
    mlp.train(X, y, epochs=1, learning_rate=1.0)


example()
