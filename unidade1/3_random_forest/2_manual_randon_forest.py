import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.utils import bike_rental_dataset

class ManualRandomForestRegressor:
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
            X_sample = X[indices]
            y_sample = y[indices]

            # Select random features
            feature_indices = self._sample_features(n_features)
            X_sample = X_sample[:, feature_indices]

            # Train tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)

            # Store the tree and the selected feature indices
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, feature_indices) in enumerate(self.trees):
            tree_predictions[:, i] = tree.predict(X[:, feature_indices])
        return np.mean(tree_predictions, axis=1)


# Carregar o dataset
df = bike_rental_dataset()
X = df.drop(['count'], axis=1).values
y = df['count'].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instanciar o RandomForestRegressor manual
manual_rf = ManualRandomForestRegressor(n_estimators=25, max_features='sqrt', max_depth=None, min_samples_split=2)

# Ajustar o modelo ao conjunto de treino
manual_rf.fit(X_train, y_train)

# Prever os rótulos do conjunto de teste
y_pred = manual_rf.predict(X_test)

# Avaliar o RMSE do conjunto de teste
rmse_test = mean_squared_error(y_test, y_pred, squared=False)

# Imprimir o RMSE do conjunto de teste
print('Test set RMSE of manual rf: {:.2f}'.format(rmse_test))
