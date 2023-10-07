# PACKAGES -------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils import load_breast_cancer_dataset


# RANDOM FOREST CLASSIFIER ---------------------------------------------------------------------------------------------

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.estimators = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):

            indices = np.random.choice(n_samples, n_samples, replace=True)  # Amostra aleatória
            X_subset = X[indices]
            y_subset = y[indices]

            # Treina a árvore
            dt = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf)
            dt.fit(X_subset, y_subset)
            self.estimators.append(dt)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.estimators]
        return np.round(np.mean(predictions, axis=0))  # classificação binária


# TESTE ----------------------------------------------------------------------------------------------------------------

# Utilizando o dataset Breast Cancer
df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]].values
y = df_breast[['diagnosis']].values

# Dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Utilizando o Random Forest
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Precisão:", accuracy)
