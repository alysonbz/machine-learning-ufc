'''Em uma atividade de casa você implementou manualmente o random forest. Esse algoritmo é exclusivo para
aplicação de variação de árvore de decisão. Implemente manualmente uma generalização, Random Classifier,
em que em vez de unicamente a aravore de decisão, o algorito possa trabalhar com o classificador KNN ou árvore de decisão.'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import numpy as np
from questao2 import data


X = data.drop('Weather Type', axis=1)
y = data['Weather Type']
class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier = 'tree', n_estimators = 100):
        self.base_classifier_type = base_classifier
        self.n_estimators = n_estimators
        self.classifier = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            if self.base_classifier_type == 'tree':
                base_clf = DecisionTreeClassifier()
            elif self.base_classifier_type == 'knn':
                base_clf = KNeighborsClassifier()
            else:
                raise ValueError("Classificador base inválido. Use 'árvore' ou 'knn'.")
            bootstrap_X, bootstrap_y = resample(X, y, replace = True, random_state=_)
            base_clf.fit(bootstrap_X, bootstrap_y)
            self.classifier.append(base_clf)

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifier])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)


random_clf = RandomClassifier(base_classifier='tree', n_estimators=100)
random_clf.fit(X, y)
predictions = random_clf.predict(X)
print(f'predições: {predictions}')