import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, label=None, n_negatives=0, n_positives=0, n=0):
        self.feature = feature
        self.label = label
        self.n_negatives = n_negatives
        self.n_positives = n_positives
        self.n = n
        self.children = {}


class HDDT:
    def __init__(self, max_depth=5, cut_off_size=1) -> None:
        self.root = None
        self.max_depth = max_depth
        self.cut_off_size = cut_off_size
        self.majority_y = None

    def count_pos_neg(self, X, y, feature, feature_value=None):
        _X = X if feature_value == None else X[X[:, feature] == feature_value]
        _y = y if feature_value == None else y[X[:, feature] == feature_value]
        pos_counts = _X[_y == 1].shape[0]
        neg_counts = _X[_y == 0].shape[0]
        return pos_counts, neg_counts

    def is_pure(self, X, y, feature):
        T_pos, T_neg = self.count_pos_neg(X, y, feature)
        return (T_pos == 0 or T_neg == 0)

    def calc_hellinger_distance(self, X, y, feature):
        f_vals = np.unique(X[:, feature])
        hellinger_value = 0
        T_pos, T_neg = self.count_pos_neg(X, y, feature)
        if T_pos == 0 or T_neg == 0:
            return 0

        for val in f_vals:
            T_v_pos, T_v_neg = self.count_pos_neg(X, y, feature, val)
            hellinger_value += (np.sqrt(T_v_pos/T_pos) -
                                np.sqrt(T_v_neg/T_neg))**2

        return np.sqrt(hellinger_value)

    def _split(self, X, y, feature):
        unique_values = np.unique(X[:, feature])
        splits = {}

        for value in unique_values:
            indices = X[:, feature] == value
            splits[value] = (X[indices], y[indices])

        return splits

    def _find_best_split(self, X, y):
        best_feature = None
        best_splits = None

        max_H = 0
        for feature in range(X.shape[1]):
            splits = self._split(X, y, feature)
            if len(splits) < 2:
                continue

            H_f = self.calc_hellinger_distance(X, y, feature)
            if H_f > max_H:
                max_H = H_f
                best_feature = feature
                best_splits = splits

        return best_feature, best_splits

    def _build_tree(self, X, y, depth=0):
        n = len(X)
        if len(np.unique(y)) == 1:
            return Node(label=y[0], n=n, n_negatives=X[y == 0].shape[0], n_positives=X[y == 1].shape[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(label=Counter(y).most_common(1)[0][0], n=n,
                        n_negatives=X[y == 0].shape[0], n_positives=X[y == 1].shape[0])

        if len(X) <= self.cut_off_size:
            return Node(label=Counter(y).most_common(1)[0][0], n=n,
                        n_negatives=X[y == 0].shape[0], n_positives=X[y == 1].shape[0])

        feature, splits = self._find_best_split(X, y)
        if feature is None or splits is None:
            return Node(label=Counter(y).most_common(1)[0][0], n=n,
                        n_negatives=X[y == 0].shape[0], n_positives=X[y == 1].shape[0])

        node = Node(feature=feature, n=n,
                    n_negatives=X[y == 0].shape[0], n_positives=X[y == 1].shape[0])
        for value, (X_subset, y_subset) in splits.items():
            child_node = self._build_tree(X_subset, y_subset, depth=depth + 1)
            node.children[value] = child_node
        return node

    def fit(self, X, y):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        self.majority_y = unique_labels[np.argmax(label_counts)]

        self.root = self._build_tree(X, y)

    def _peredic_single_probs(self, x, node):
        if node.label is not None:
            return [node.n_negatives/node.n, node.n_positives/node.n]

        feature_value = x[node.feature]
        if feature_value in node.children:
            return self._peredic_single_probs(x, node.children[feature_value])
        else:
            return [node.n_negatives/node.n, node.n_positives/node.n]

    def predict_prob(self, X):
        return [self._peredic_single_probs(x, self.root) for x in X]

    def _predict_single(self, x, node):
        if node.label is not None:
            return node.label

        feature_value = x[node.feature]
        if feature_value in node.children:
            return self._predict_single(x, node.children[feature_value])
        else:
            return np.argmax([node.n_negatives/node.n, node.n_positives/node.n])
            # return self.majority_y

    def predict(self, X):
        return [self._predict_single(x, self.root) for x in X]
