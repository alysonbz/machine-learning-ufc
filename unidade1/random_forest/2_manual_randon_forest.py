import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.utils import bike_rental_dataset

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
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[indices]
            y_subset = y[indices]
            dt = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf)
            dt.fit(X_subset, y_subset)
            self.estimators.append(dt)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.estimators]
        return np.round(np.mean(predictions, axis=0))

# Carregue o conjunto de dados de aluguel de bicicletas
df = bike_rental_dataset()

# Remova a coluna "count" que é o alvo de regressão
X = df.drop(['count'], axis=1).values
y = df['count'].values

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Use o Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Arredonde as previsões para inteiros
y_pred = np.round(y_pred)

# Calcule a precisão da classificação
accuracy = accuracy_score(y_test, y_pred)

print("Precisão:", accuracy)
