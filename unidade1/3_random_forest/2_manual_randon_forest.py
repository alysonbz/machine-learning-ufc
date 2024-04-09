import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import bike_rental_dataset
from sklearn.model_selection import train_test_split

# Define helper functions for manual random forest

def bootstrap_sample(X, y):
    # Randomly sample with replacement
    indices = np.random.choice(len(X), size=len(X), replace=True)
    return X[indices], y[indices]

def random_split(X, y):
    # Randomly select feature for splitting
    feature_index = np.random.choice(X.shape[1])
    # Randomly select split point
    split_value = np.random.choice(X[:, feature_index])
    # Split data
    X_left = X[X[:, feature_index] <= split_value]
    X_right = X[X[:, feature_index] > split_value]
    y_left = y[X[:, feature_index] <= split_value]
    y_right = y[X[:, feature_index] > split_value]
    return (X_left, y_left), (X_right, y_right), feature_index, split_value

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions / len(self.trees)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(X) < self.min_samples_split or len(set(y)) == 1:
            return {'prediction': np.mean(y)}
        else:
            left, right, feature_index, split_value = random_split(X, y)
            return {'feature_index': feature_index,
                    'split_value': split_value,
                    'left': self._build_tree(*left, depth + 1),
                    'right': self._build_tree(*right, depth + 1)}

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if 'prediction' in tree:
            return tree['prediction']
        else:
            if x[tree['feature_index']] <= tree['split_value']:
                return self._predict_tree(x, tree['left'])
            else:
                return self._predict_tree(x, tree['right'])

# Load data
df = bike_rental_dataset()
X = df.drop(['count'], axis=1).values
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate rf
rf = RandomForestRegressor(n_estimators=100)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Manual random forest doesn't directly provide feature importances

# Plotting importances won't be possible in this case
