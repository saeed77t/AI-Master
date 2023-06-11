# %%
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from HDDT import HDDT
from utils import perform_grid_search

# %%


def select_minority_vs_rest(X, y):
    y = np.array(y)

    unique_labels, label_counts = np.unique(y, return_counts=True)
    minority_class = unique_labels[np.argmin(label_counts)]

    binary_labels = np.where(y == minority_class, 1, 0)

    return X, binary_labels

# %%
# data = pd.read_csv("Covid19HDDT.csv")
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# X_minor, y_minor = separate_minority_class(X, y)
# new_data = np.hstack((X_minor, y_minor.reshape((-1, 1))))
# data = pd.DataFrame(new_data)

# correlations = np.array(data.corrwith(data.iloc[:, -1], method="kendall"))[:-1]
# print(f"Correlations with TARGET:\n", data.corrwith(data.iloc[:, -1], method="kendall"))

# %% [markdown]
# ### Reading data


# %%
data = np.array(pd.read_csv("Covid19HDDT.csv"))
X = data[:, :-1]
y = data[:, -1]

# %% [markdown]
# ### Removing high correlations (maximum hellinger distances)

# %%


def calc_hellinger_distance(X, y, feature):
    f_vals = np.unique(X[:, feature])
    hellinger_value = 0

    for val in f_vals:
        hellinger_value += (np.sqrt(X[(X[:, feature] == val) & (y == 1)].shape[0]/X[y == 1].shape[0]) -
                            np.sqrt(X[(X[:, feature] == val) & (y == 0)].shape[0]/X[y == 0].shape[0]))**2

    return np.sqrt(hellinger_value)


h_dists = []
X_minor, y_minor = select_minority_vs_rest(X, y)
for feature in range(X_minor.shape[1]):
    h_dists.append(calc_hellinger_distance(X_minor, y_minor, feature))

selected = np.where(np.array(h_dists) < 1)[0]
X = X[:, selected]

# %%


def undersample(X, y):
    values, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)

    X_new = None
    y_new = None
    for i, v in enumerate(values):
        idxs = np.random.choice(np.where(y == v)[0], min_samples)
        if(i == 0):
            X_new = X[idxs]
            y_new = y[idxs]
        else:
            X_new = np.concatenate((X_new, X[idxs]), axis=0)
            y_new = np.concatenate((y_new, y[idxs]), axis=0)
    return X_new, y_new

# %% [markdown]
# ### Minority vs Rest


# %%

X_minor, y_minor = X_minor, y_minor = select_minority_vs_rest(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_minor, y_minor, stratify=y_minor, test_size=0.3, random_state=2)

classifier = HDDT()
param_grid = {'max_depth': [2, 3, 4, 5, None], 'cut_off_size': [10, 50, 100]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, print_std=False, n_iterations=10)

# %% [markdown]
# ### One vs. One

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=2)

# %%
# Function to find the most common element in a 1D array


def find_most_common(column):
    non_none_elements = column[column != None]
    unique_elements, counts = np.unique(non_none_elements, return_counts=True)
    if len(unique_elements) == 0:
        return None
    most_common_index = np.argmax(counts)
    return unique_elements[most_common_index]

# %%


class OVO_HDDT:
    def __init__(self, max_depth=2, cut_off_size=1) -> None:
        self.models = []
        self.max_depth = max_depth
        self.cut_off_size = cut_off_size

    def select_OVO(self, X, y, l1, l2):
        _X = X[(y == l1) | (y == l2), :]
        _y = y[(y == l1) | (y == l2)]
        _y[_y == l1] = 0
        _y[_y == l2] = 1

        return _X, _y

    def fit(self, X, y):
        classes = np.unique(y)
        self.models = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                # print(f"Class {i} vs. {j}")
                # X_train_, y_train_ = undersample(X_train, y_train)
                _X, _y = self.select_OVO(X, y, i, j)

                hddt = HDDT(max_depth=self.max_depth,
                            cut_off_size=self.cut_off_size)
                hddt.fit(_X, _y)
                self.models.append(([i, j], hddt))

    def predict(self, X):
        predictions = np.array([])
        for i, model in enumerate(self.models):
            y_preds = np.array(model[1].predict(X))
            cls_0 = np.where(y_preds == 0)[0]
            cls_1 = np.where(y_preds == 1)[0]
            y_preds[cls_0] = model[0][0]
            y_preds[cls_1] = model[0][1]

            if(i == 0):
                predictions = y_preds
            else:
                predictions = np.vstack((predictions, y_preds))

        y_pred = np.apply_along_axis(find_most_common, axis=0, arr=predictions)

        return y_pred


# %%
classifier = OVO_HDDT()
param_grid = {'max_depth': [2, 3, 4, 5], 'cut_off_size': [10, 50, 100]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, print_std=False, n_iterations=10)

# %% [markdown]
# ### One vs. All

# %%


class OVA_HDDT:
    def __init__(self, max_depth=2, cut_off_size=1) -> None:
        self.models = []
        self.max_depth = max_depth
        self.cut_off_size = cut_off_size

    def select_OVA(self, X, y, l):
        _y = np.select([y == l, y != l], [1, 0], y)
        return X, _y

    def fit(self, X, y):
        classes = np.unique(y)
        self.models = []
        for i in range(len(classes)):
            # print(f"Class {i} vs. All")
            # X_train_, y_train_ = undersample(X_train, y_train)
            _X, _y = self.select_OVA(X, y, i)

            hddt = HDDT(max_depth=self.max_depth,
                        cut_off_size=self.cut_off_size)
            hddt.fit(_X, _y)
            self.models.append((i, hddt))

    def predict(self, X):
        predictions = np.array([])
        for i, model in enumerate(self.models):
            y_pred_probs = model[1].predict_prob(X)
            y_pred_probs = [pred[1] for pred in y_pred_probs]

            if(i == 0):
                predictions = y_pred_probs
            else:
                predictions = np.vstack((predictions, y_pred_probs))

        y_pred = np.apply_along_axis(np.argmax, axis=0, arr=predictions)

        return y_pred


# %%
classifier = OVA_HDDT()
param_grid = {'max_depth': [2, 3, 4, 5], 'cut_off_size': [10, 50, 100]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, print_std=False, n_iterations=10)

# %%
