import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class RandomClassifier:
    def __init__(self, classifier):
        """
        Inicializa o RandomClassifier com um classificador.

        :param classifier: Um objeto classificador do scikit-learn.
        """
        self.classifier = classifier

    def fit(self, X_train, y_train):
        """
        Treina o classificador com os dados fornecidos.

        :param X_train: Dados de treinamento.
        :param y_train: Rótulos de treinamento.
        """
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Faz previsões usando o classificador treinado.

        :param X_test: Dados de teste.
        :return: Previsões feitas pelo classificador.
        """
        return self.classifier.predict(X_test)


# Carregar o dataset
dataset = pd.read_csv('dataset.csv')

# Remover colunas desnecessárias
data_cleaned = dataset.drop(columns=['Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
                                     'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13',
                                     'Symptom_14', 'Symptom_15', 'Symptom_16', 'Symptom_17'])

# Criar dummies para as variáveis categóricas
data_dummies = pd.get_dummies(data_cleaned, drop_first=True)

# Identificar as colunas que são alvos
target_columns = [col for col in data_dummies.columns if col.startswith('Disease_')]
X = data_dummies.drop(columns=target_columns)
y = data_dummies[target_columns]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Testar DecisionTreeClassifier
dt_classifier = RandomClassifier(DecisionTreeClassifier(random_state=42))
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Avaliar DecisionTreeClassifier
y_test_single = y_test.idxmax(axis=1)
y_pred_single_dt = pd.DataFrame(y_pred_dt, columns=target_columns).idxmax(axis=1)

print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test_single, y_pred_single_dt))
print("Classification Report:")
print(classification_report(y_test_single, y_pred_single_dt))

# Testar KNeighborsClassifier
knn_classifier = RandomClassifier(KNeighborsClassifier())
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

# Avaliar KNeighborsClassifier
y_pred_single_knn = pd.DataFrame(y_pred_knn, columns=target_columns).idxmax(axis=1)

print("\nK-Nearest Neighbors Classifier:")
print("Accuracy:", accuracy_score(y_test_single, y_pred_single_knn))
print("Classification Report:")
print(classification_report(y_test_single, y_pred_single_knn))


