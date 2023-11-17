from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import numpy as np


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier='tree', n_estimators=100):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Criar e ajustar classificadores base
            if self.base_classifier == 'tree':
                base_clf = DecisionTreeClassifier()
            elif self.base_classifier == 'knn':
                base_clf = KNeighborsClassifier()
            else:
                raise ValueError("Invalid base classifier. Use 'tree' or 'knn'.")

            # Criar conjuntos de dados de inicialização aleatória para cada classificador
            bootstrap_X, bootstrap_y = resample(X, y, replace=True, random_state=_)

            # Treinar classificador base com o conjunto de dados de inicialização
            base_clf.fit(bootstrap_X, bootstrap_y)
            self.classifiers.append(base_clf)

    def predict(self, X):
        # Fazer previsões com cada classificador base
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        # Maioria de votos para a previsão final
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)


RandomClassifier.fit(X)