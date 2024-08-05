import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class ManualRandomClassifier:
    def __init__(self, base_classifier='decision_tree', n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, n_neighbors=5):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_neighbors = n_neighbors
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
            y_sample = y.iloc[indices]  # Use iloc for row selection

            # Select random features
            feature_indices = self._sample_features(n_features)
            X_sample = X_sample.iloc[:, feature_indices]

            # Train classifier
            if self.base_classifier == 'decision_tree':
                classifier = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            elif self.base_classifier == 'knn':
                classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            else:
                raise ValueError("Unsupported classifier type")

            classifier.fit(X_sample, y_sample)

            # Store the classifier and the selected feature indices
            self.trees.append((classifier, feature_indices))

    def predict(self, X):
        tree_predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        for i, (classifier, feature_indices) in enumerate(self.trees):
            tree_predictions[:, i] = classifier.predict(X.iloc[:, feature_indices])

        # Use majority voting for the final prediction
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_predictions)
        return final_predictions

    def train_and_evaluate(self, X, y, X_test, y_test):
        # Apply SMOTE to balance the classes in the training set
        smote = SMOTE(random_state=1)
        X_train_balanced, y_train_balanced = smote.fit_resample(X, y)

        # Train the classifier
        self.fit(X_train_balanced, y_train_balanced)

        # Make predictions
        predictions = self.predict(X_test)

        # Evaluate accuracy
        acc = np.sum(y_test == predictions) / len(y_test)
        return acc

# Carregando e processando os dados
df = pd.read_csv("db_ajustado.csv")

# Selecionando as colunas mais adequadas
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = df['quality']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Testando com DecisionTreeClassifier
clf_tree = ManualRandomClassifier(base_classifier='decision_tree', n_estimators=100, max_features='sqrt', max_depth=6)
acc_tree = clf_tree.train_and_evaluate(X_train, y_train, X_test, y_test)
print(f"Accuracy with Decision Tree: {acc_tree}")

# Testando com KNeighborsClassifier
clf_knn = ManualRandomClassifier(base_classifier='knn', n_estimators=100, max_features='sqrt', n_neighbors=5)
acc_knn = clf_knn.train_and_evaluate(X_train, y_train, X_test, y_test)
print(f"Accuracy with KNN: {acc_knn}")
