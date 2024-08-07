import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('alzheimers_disease_data.csv')

# Exclusão de dados inúteis
df = df.drop('DoctorInCharge', axis=1)
df = df.drop(['Confusion', 'HeadInjury', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Hypertension',
              'Disorientation', 'Smoking', 'FamilyHistoryAlzheimers', 'Forgetfulness', 'Gender',
              'CardiovascularDisease', 'Depression', 'Diabetes'], axis=1)

# Preprocessing
# Suponha que a coluna alvo seja 'Diagnosis' e que as colunas restantes sejam características
X = df.drop(columns=['Diagnosis'])  # Excluir a coluna alvo
y = df['Diagnosis']  # Coluna alvo

# Certificar que X é do tipo float
X = X.astype(float)

# Escalonamento dos dados para KNN
def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_scaled = standard_scaler(X)

# Dividir o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Implementação manual do K-Nearest Neighbors
class KNearestNeighbors:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]

# Criar a classe RandomClassifier
class RandomClassifier:
    def __init__(self, base_classifier='knn', n_estimators=10, knn_neighbors=5):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.knn_neighbors = knn_neighbors
        self.models = []

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X_resampled, y_resampled = X.iloc[indices], y.iloc[indices]

            # Initialize the base classifier
            if self.base_classifier == 'knn':
                model = KNearestNeighbors(n_neighbors=self.knn_neighbors)
            else:
                raise ValueError("Base classifier must be 'knn'")

            # Fit the model
            model.fit(X_resampled, y_resampled)
            self.models.append(model)
        return self

    def predict(self, X):
        X = np.array(X)
        # Collect predictions from each model
        predictions = np.zeros((X.shape[0], self.n_estimators), dtype=int)
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)

        # Use majority voting to decide final prediction
        final_predictions = [Counter(predictions[i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final_predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Criar e treinar o classificador com KNN
clf_knn = RandomClassifier(base_classifier='knn', n_estimators=50, knn_neighbors=5)
clf_knn.fit(X_train, y_train)

# Fazer previsões e avaliar
y_pred_knn = clf_knn.predict(X_test)
print("Acurácia com KNN:", accuracy_score(y_test, y_pred_knn))
print("Relatório de Classificação com KNN:")
print(classification_report(y_test, y_pred_knn))
print("Matriz de Confusão com KNN:")
print(confusion_matrix(y_test, y_pred_knn))

def plot_confusion_matrix(cm, title='Matriz de Confusão'):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

cm_knn = confusion_matrix(y_test, y_pred_knn)
plot_confusion_matrix(cm_knn, title='Matriz de Confusão - KNN')
# Criar e treinar a árvore de decisão
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo de Árvore de Decisão
y_pred_tree = clf_tree.predict(X_test)
print("\nAcurácia com Árvore de Decisão:", accuracy_score(y_test, y_pred_tree))
print("Relatório de Classificação com Árvore de Decisão:")
print(classification_report(y_test, y_pred_tree))
print("Matriz de Confusão com Árvore de Decisão:")
print(confusion_matrix(y_test, y_pred_tree))




