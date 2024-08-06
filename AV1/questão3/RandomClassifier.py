import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

class ManualRandomClassifier:
    def __init__(self, base_classifier='decision_tree', n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, n_neighbors=8):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_neighbors = n_neighbors
        self.trees = []

    def sample_features(self, n_features):
        if self.max_features == 'sqrt':
            # Seleciona uma quantidade de características igual à raiz quadrada do total
            return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif isinstance(self.max_features, int):
            # Seleciona um número fixo de características
            return np.random.choice(n_features, self.max_features, replace=False)
        else:
            # Usa todas as características disponíveis
            return np.arange(n_features)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            # Amostragem com reposição
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]

            # Seleciona características aleatórias
            feature_indices = self.sample_features(n_features)
            X_sample = X_sample.iloc[:, feature_indices]

            # Treina o classificador
            if self.base_classifier == 'decision_tree':
                classifier = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            elif self.base_classifier == 'knn':
                classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            else:
                raise ValueError("Tipo de classificador não suportado")

            classifier.fit(X_sample, y_sample)
            # Armazena o classificador e os índices das características selecionadas
            self.trees.append((classifier, feature_indices))

    def predict(self, X):
        tree_predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        for i, (classifier, feature_indices) in enumerate(self.trees):
            # Predições de cada árvore
            tree_predictions[:, i] = classifier.predict(X.iloc[:, feature_indices])

        # Votação majoritária para a previsão final
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_predictions)
        return final_predictions

    def train_evaluate(self, X, y, X_test, y_test):
        # Aplica SMOTE para balancear as classes no conjunto de treinamento
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X, y)

        # Treina o classificador
        self.fit(X_train_balanced, y_train_balanced)

        # Faz previsões
        predictions = self.predict(X_test)

        # Avalia a acurácia
        acc = np.sum(y_test == predictions) / len(y_test)
        return acc
