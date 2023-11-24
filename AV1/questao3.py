import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Carregando o conjunto de dados
data = pd.read_csv(r"C:\Users\laura\Downloads\winequality-red.csv")

# Separando em atributos (X) e rótulos (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementando RandomClassifier para escolher entre DecisionTreeClassifier e KNeighborsClassifier
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, classifier="decision_tree"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.estimators = []
        self.classifier = classifier

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X.iloc[indices]
            y_subset = y.iloc[indices]

            if self.classifier == "decision_tree":
                classifier = DecisionTreeClassifier(max_depth=self.max_depth,
                                                    min_samples_split=self.min_samples_split,
                                                    min_samples_leaf=self.min_samples_leaf)
            elif self.classifier == "knn":
                classifier = KNeighborsClassifier()
            else:
                raise ValueError("Classifier not supported")

            classifier.fit(X_subset, y_subset)
            self.estimators.append(classifier)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.estimators]
        return np.round(np.mean(predictions, axis=0))  # classificação binária

# amostragem aleatória
def random_indices(n_samples, sample_size):
    return np.random.choice(n_samples, sample_size, replace=False)

# Número de amostras para treinamento
sample_size = len(X_train)

# Amostra aleatória de índices
indices = random_indices(len(X_train), sample_size)

# Subconjunto de treinamento
X_train_subset = X_train.iloc[indices]
y_train_subset = y_train.iloc[indices]
# _______________________________________________________________________________________________________________________
# Teste com a Árvore de Decisão
random_classifier_dt = RandomForestClassifier(classifier="decision_tree")
random_classifier_dt.fit(X_train_subset, y_train_subset)
y_pred_dt = random_classifier_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Acurácia da Árvore de Decisão:", accuracy_dt)

# _______________________________________________________________________________________________________________________

# Teste com o KNN
random_classifier_knn = RandomForestClassifier(classifier="knn")
random_classifier_knn.fit(X_train_subset, y_train_subset)
y_pred_knn = random_classifier_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Acurácia do KNN:", accuracy_knn)
