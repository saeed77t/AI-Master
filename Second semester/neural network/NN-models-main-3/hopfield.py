import numpy as np
from tabulate import tabulate


class hopfield:
    def __init__(self, input_size: int, printing: bool = False):
        self.input_size = input_size
        self.w = np.zeros((input_size, input_size))
        self.printing = printing

    def train(self, input_data: np.ndarray):
        """
        Get input data which is include p patterns.
        """
        p = input_data.shape[0]
        # for i in range(self.input_size):
        #     for j in range(i + 1, self.input_size):
        #         for k in range(p):
        #             self.w[i][j] += input_data[k][i] * input_data[k][j]
        #         self.w[j][i] = self.w[i][j]

        # wp = np.zeros((self.input_size, self.input_size))
        for i in range(p):
            self.w += np.outer(input_data[i], input_data[i])
        self.w -= np.eye(self.input_size) * p
        if self.printing:
            self.print_weight()

    def recall_synchrouns(self, input_data: np.ndarray, max_iter: int = 100):
        """
        Recall the input data.
        """
        if self.printing:
            print(f"\n\ny(0) = {input_data})")
            print(f"energy = {self.energy(input_data)}")
        output_data = input_data.copy()
        for i in range(max_iter):
            output_data = input_data + np.dot(self.w, output_data)
            output_data[output_data >= 0] = 1
            output_data[output_data < 0] = -1
            if self.printing:
                print(f"\ny({i + 1}) = {output_data}")
                print(f"energy = {self.energy(output_data)}")
            if np.all(input_data == output_data):
                break

        return output_data

    def recall_asynchrouns(self, X: np.ndarray, max_iter: int = 100):
        """
        Recall the input data with asynchrouns update.
        """
        Y = X.copy()

        for i in range(max_iter):
            if self.printing:
                print(f"\n\ny({i}) = {Y})")
                print(f"energy = {self.energy(Y)}")
            pre_Y = Y.copy()
            rand_index = np.random.permutation(self.input_size)
            if self.printing:
                print(f"\nrandom index: {rand_index}")
            for j in rand_index:
                Y[j] = np.sign(X[j] + self.w.T[j] @ Y)
                if self.printing:
                    print(f"\nselect {j}th neuron")
                    print(f"y({i}) = {Y}")
                    print(f"energy = {self.energy(Y)}")
            if np.all(pre_Y == Y):
                print(f"Converged at {i+1}th iteration\n")
                break

        return Y

    def print_weight(self):
        print("Weight matrix:")
        print(tabulate(self.w, tablefmt="fancy_grid"))

    def energy(self, input_data: np.ndarray):
        """
        Calculate the energy of the input data.
        """
        e = 0
        for i in range(self.input_size):
            for j in range(self.input_size):
                e += -self.w[i][j] * input_data[i] * input_data[j]
        return e / 2


def test_hopfield():
    S = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]])

    model = hopfield(input_size=len(S[0]), printing=True)
    model.train(S)

    sample = np.array([1, 1, 0, -1])
    print(f"Input: {sample}")
    print(f"Output: {model.recall_synchrouns(sample)}")
    print(f"Output: {model.recall_asynchrouns(sample)}")


def test_slide_19():
    """
    The output of synchrouns and asynchrouns method is different.
    """
    S = np.array([[1, 1, -1, -1]])

    model = hopfield(input_size=len(S[0]), printing=True)
    model.train(S)

    sample = np.array([-1, -1, -1, -1])
    print(f"Input: {sample}")
    print(f"Output: {model.recall_synchrouns(sample)}")
    print(f"Output: {model.recall_asynchrouns(sample)}")  # Got in loop :(


def example3():
    s1 = [1, -1, -1, 1]
    s2 = [-1, -1, 1, 1]
    s3 = [-1, -1, -1, -1]

    S = np.array([s1, s2, s3])

    model = hopfield(input_size=len(S[0]), printing=True)
    model.train(S)

    sample1 = [-1, 0, 0, 1]
    model.recall_asynchrouns(sample1)
    # model.recall_synchrouns(sample1) # Got in loop :(


def midterm_p9():
    s1 = [-1, 1, -1]
    s2 = [1, -1, -1]
    s3 = [1, 1, 1]

    sample = [-1, -1, 0]

    S = np.array([s1, s2, s3])

    model = hopfield(input_size=S.shape[1], printing=True)
    model.train(S)

    model.recall_synchrouns(sample)


midterm_p9()

# example3()
# test_hopfield()
# test_slide_19()
