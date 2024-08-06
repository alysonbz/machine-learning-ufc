import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from src.utils import bike_rental_dataset


class ManualRandomForestClassifier:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def _sample_features(self, n_features):
        if self.max_features == 'sqrt':
            return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif isinstance(self.max_features, int):
            return np.random.choice(n_features, self.max_features, replace=False)
        else:
            return np.arange(n_features)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X.iloc[indices]  # Use iloc for row selection
            y_sample = y[indices]

            # Select random features
            feature_indices = self._sample_features(n_features)
            X_sample = X_sample.iloc[:, feature_indices]

            # Train
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)

            # Store the tree and the selected feature indices
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        for i, (tree, feature_indices) in enumerate(self.trees):
            tree_predictions[:, i] = tree.predict(X.iloc[:, feature_indices])

        # Use majority voting for the final prediction
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_predictions)
        return final_predictions


data = bike_rental_dataset()

X = data.drop(['count'], axis=1)
y = data['count'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

clf = ManualRandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=6)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)
print(acc)