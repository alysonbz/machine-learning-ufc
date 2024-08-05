import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class RandomClassifier:
    def __init__(self, n_classifiers=10, classifier_type='decision_tree', k_neighbors=3):
        self.n_classifiers = n_classifiers
        self.classifier_type = classifier_type
        self.k_neighbors = k_neighbors
        self.classifiers = []

    def fit(self, X, y):
        y = np.ravel(y)
        self.classifiers = []
        for _ in range(self.n_classifiers):
            if self.classifier_type == 'decision_tree':
                clf = DecisionTreeClassifier()
            elif self.classifier_type == 'knn':
                clf = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            else:
                raise ValueError("Tipo de classificador Invalido. Use 'decision_tree' or 'knn'.")

            sample_indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            clf.fit(X_sample, y_sample)
            self.classifiers.append(clf)

    def predict(self, X):
        predictions = np.zeros((len(X), self.n_classifiers))
        for idx, clf in enumerate(self.classifiers):
            predictions[:, idx] = clf.predict(X)

        # Majority voting
        final_predictions = [np.argmax(np.bincount(predictions[i].astype(int))) for i in range(len(X))]
        return np.array(final_predictions)


# Carregar os dados
df = pd.read_csv("/home/ufc/savim/machine-learning-ufc/AV1/plant_growth_data_pos.csv")

X = df.drop(['Growth_Milestone'], axis=1).values
y = df[['Growth_Milestone']].values

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Criar e treinar o Random Classifier_rf
random_classifier = RandomClassifier(n_classifiers=10, classifier_type='decision_tree', k_neighbors=3)
random_classifier.fit(X_train, y_train)

# Criar e treinar o Random Classifier_knn
random_classifier_knn = RandomClassifier(n_classifiers=10, classifier_type='knn', k_neighbors=3)
random_classifier_knn.fit(X_train, y_train)

# Fazer previsões
y_pred = random_classifier.predict(X_test)
y_pred_knn = random_classifier_knn.predict(X_test)

# Avaliar o desempenho
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Decision Tree: {accuracy}')

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Acurracy knn: {accuracy_knn}')
