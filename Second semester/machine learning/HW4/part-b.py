# %%
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
import pandas as pd
from sklearn.utils import resample
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from HDDT import HDDT
from utils import perform_grid_search
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%


class BaggingWithUndersampling:
    def __init__(self, n_estimators=10, samples_ceof=1.0, base_learner=DecisionTreeClassifier(), disable_under_sampling=False):
        self.n_estimators = n_estimators
        self.samples_ceof = samples_ceof
        self.base_learner = base_learner
        self.estimators = []
        self.disable_under_sampling = disable_under_sampling

    def fit(self, X, y):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        minority_label = unique_labels[np.argmin(label_counts)]

        minority_class_indices = np.where(y == minority_label)[0]
        majority_class_indices = np.where(y == majority_label)[0]

        for _ in range(self.n_estimators):
            base_estimator = self.base_learner
            if(self.disable_under_sampling):
                X_subset, y_subset = X, y
            else:
                majority_class_indices_sampled = np.random.choice(majority_class_indices,
                                                                  int(len(
                                                                      minority_class_indices) * self.samples_ceof),
                                                                  replace=False)
                indices = np.concatenate(
                    (minority_class_indices, majority_class_indices_sampled))
                X_subset, y_subset = X[indices], y[indices]

            base_estimator.fit(X_subset, y_subset)
            self.estimators.append(base_estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X)
                               for estimator in self.estimators], dtype=int)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)


# %%


class AdaBoostUnderSampling:
    def __init__(self, base_learner, n_estimators=10, rounds=10):
        self.n_estimators = n_estimators
        self.rounds = rounds
        self.base_learner = base_learner
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        minority_label = unique_labels[np.argmin(label_counts)]

        minority_class_indices = np.where(y == minority_label)[0]
        majority_class_indices = np.where(y == majority_label)[0]

        n = X.shape[0]

        for _ in range(self.n_estimators):
            weights = np.ones(n) / n

            for __ in range(self.rounds):
                majority_class_indices_sampled = np.random.choice(majority_class_indices, len(minority_class_indices),
                                                                  replace=False)
                indices = np.concatenate(
                    (minority_class_indices, majority_class_indices_sampled))
                X_subset, y_subset = X[indices], y[indices]

                self.base_learner.fit(X_subset, y_subset)

                predictions = self.base_learner.predict(X)
                error = np.sum(weights[y != predictions])

                alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

                weights *= np.exp(alpha * (predictions != y))
                weights /= np.sum(weights)

            self.estimators.append(self.base_learner)
            self.alphas.append(alpha)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        for model, alpha in zip(self.estimators, self.alphas):
            predictions += np.array(alpha, dtype=np.float64) * model.predict(X)

        return np.sign(predictions)


# %%

data = np.array(pd.read_csv("Covid.csv"))
X = data[:, :-1]
y = data[:, -1]
cls_0 = np.where(y == -1)[0]
cls_1 = np.where(y == 1)[0]
y[cls_0] = 0
y[cls_1] = 1

unique_labels, label_counts = np.unique(y, return_counts=True)
minority_label = unique_labels[np.argmin(label_counts)]

# %%

imp = IterativeImputer(max_iter=300, random_state=1,
                       initial_strategy="most_frequent")
imp.fit(X)
X = np.round(imp.transform(X), 1)

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=2)

# %% [markdown]
# ## Bagging

# %% [markdown]
# ### Built-in Decision tree

# %%
learner = DecisionTreeClassifier()
classifier = BaggingWithUndersampling(base_learner=learner)
param_grid = {'n_estimators': [11, 31, 51, 101]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, n_iterations=10, plot_roc=True)

# %% [markdown]
# ### HDDT

# %% [markdown]
# #### With under-sampling

# %%
learner = HDDT(max_depth=25, cut_off_size=10)
classifier = BaggingWithUndersampling(base_learner=learner)
param_grid = {'n_estimators': [11, 31, 51, 101]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, n_iterations=10, plot_roc=True)

# %% [markdown]
# #### Without under-sampling

# %%
learner = HDDT(max_depth=25, cut_off_size=5)
classifier = BaggingWithUndersampling(
    base_learner=learner, disable_under_sampling=True)
param_grid = {'n_estimators': [11, 31, 51, 101]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, n_iterations=2, plot_roc=True)

# %% [markdown]
# ## AdaBoost

# %%
learner = DecisionTreeClassifier()
classifier = AdaBoostUnderSampling(base_learner=learner)
param_grid = {'n_estimators': [11, 31, 51, 101], 'rounds': [10, 15]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, n_iterations=10, plot_roc=True)

# %%
learner = HDDT(max_depth=12, cut_off_size=10)
classifier = AdaBoostUnderSampling(base_learner=learner)
param_grid = {'n_estimators': [11, 31, 51, 101], 'rounds': [10, 15]}

# Perform grid search
perform_grid_search(classifier, param_grid, X_train, y_train,
                    X_test, y_test, n_iterations=10, plot_roc=True)
