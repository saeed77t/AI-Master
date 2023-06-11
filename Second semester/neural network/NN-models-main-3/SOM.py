import numpy as np
from tabulate import tabulate


class SOM:
    def __init__(
        self,
        input_size,
        height,
        width,
        r=0,
        a=0.2,
        weight=None,
        eps=0.1,
        max_iter=10,
        printing=False,
    ):
        """
        SOM with 2D grid(height, width), learning rate a, and radius r
        """
        self.height = height
        self.width = width
        self.r = r
        self.a = a
        if weight is None:
            self.weight = np.random.rand(height, width, input_size)
        else:
            self.weight = weight
        self.printing = printing
        self.eps = eps
        self.max_iter = max_iter
        self.table = np.empty((0, 2 * height * width + 3))
        self.decimal_point = 2

    def train(self, X: np.ndarray):
        """
        Get input data and train the SOM
        """
        pre_weight = self.weight.copy()

        if self.printing:
            print(f"\nweight(0) = {self.weight}")
        for i in range(self.max_iter):
            self.table = np.vstack(
                (self.table, [f"epo {i+1}"] * self.table.shape[1])
            )
            for x in X:
                if self.printing:
                    row = [
                        self.weight[j][k]
                        for j in range(self.height)
                        for k in range(self.width)
                    ]
                # Find the best matching unit
                bmu, dis = self.find_bmu(x)

                # Update the weight of the BMU and its neighbors
                self.update_weight(bmu, x)

                if self.printing:
                    row.append(x)
                    for i in range(len(dis)):
                        row.append(dis[i])
                    row.append(bmu)
                    row.append(
                        self.weight[bmu[0]][bmu[1]]
                        - pre_weight[bmu[0]][bmu[1]]
                    )  # Δw
                    self.table = np.vstack((self.table, row))

                if self.printing:
                    print(f"\nweight({i+1}) = {self.weight}")
            if np.all(np.abs(pre_weight - self.weight) < self.eps):
                break
            pre_weight = self.weight.copy()

        return self.weight

    def find_bmu(self, x: np.ndarray):
        """
        Find the best matching unit
        """
        bmu = None
        min_dist = np.inf
        dis = []
        for i in range(self.height):
            for j in range(self.width):
                dist = np.linalg.norm(x - self.weight[i][j])
                dis.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    bmu = (i, j)
        return bmu, np.round(dis, self.decimal_point)

    def update_weight(self, bmu, x):
        """
        Update the weight of the BMU and its neighbors
        """
        for i in range(self.height):
            for j in range(self.width):
                dist = np.max(abs(np.array(bmu) - np.array([i, j])))
                if dist <= self.r:
                    self.weight[i][j] += (
                        self.a
                        * (x - self.weight[i][j])
                        # * np.exp(-(dist**2) / (2 * self.r**2))
                    )

        self.weight = np.round(self.weight, self.decimal_point)

    def print_table(self):
        label = []
        for i in range(self.height):
            for j in range(self.width):
                label.append(f"w({i+1}, {j+1})")

        label.append("x")

        for i in range(self.height):
            for j in range(self.width):
                label.append(f"||x-w{i+1, j+1}||")

        label.append("k")
        label.append("Δw")

        print(tabulate(self.table, headers=label, tablefmt="fancy_grid"))
        pass


def test_som_linear_winner_takes_all():
    """
    example of slide 27 of ch2.4
    """
    S = np.array([[1, 1], [-1, 1], [1, 0.5], [-0.5, 1]])
    init_weight = np.array([[[0.1, 0.3], [0.4, 0.5], [0.3, 0.4]]])
    som = SOM(
        input_size=S.shape[0],
        height=1,
        width=3,
        weight=init_weight,
        r=0,
        a=0.2,
        printing=True,
    )

    som.train(S)

    som.print_table()


def example2():
    S = np.array([[0, 0], [0, 1], [2, 2], [2, 0], [2, 1], [0, 2]])
    L = np.array([1, 1, 1, 2, 2, 2])

    model = SOM(input_size=S.shape[1], height=1, width=2, r=0, printing=True)
    model.train(S)
    model.print_table()


example2()
# test_som_linear_winner_takes_all()
