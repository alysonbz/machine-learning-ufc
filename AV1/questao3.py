import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Carregar os dados
data = pd.read_csv("Skyserver.csv")  # Altere para o caminho correto do arquivo CSV

# Pré-processamento dos dados
X = data.drop('class', axis=1)
y = data['class']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definição da classe RandomClassifier
class RandomClassifier:
    def __init__(self, models):
        self.models = models
        self.selected_model = None

    def fit(self, X, y):
        # Seleciona aleatoriamente um modelo para ser usado
        self.selected_model = np.random.choice(self.models)
        self.selected_model.fit(X, y)

    def predict(self, X):
        return self.selected_model.predict(X)

# Lista de modelos possíveis
models = [
    DecisionTreeClassifier(),
    KNeighborsClassifier()
]

# Treinamento e avaliação
rc = RandomClassifier(models)
rc.fit(X_train, y_train)
y_pred = rc.predict(X_test)
print("Relatório de Classificação (Random Classifier):\n", classification_report(y_test, y_pred))
