import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Carregue o conjunto de dados
data = load_breast_cancer()

# Separe os dados em X (características) e y (rótulos)
X = data.data
y = data.target

# Divida o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Defina uma faixa de profundidades para testar
depths = np.arange(1, 10)
biases = []
variances = []

# Calcule o bias e a variância para diferentes profundidades da árvore
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Calcule o bias (erro de treinamento) e a variância (erro de teste)
    bias = 1 - model.score(X_train, y_train)
    variance = 1 - model.score(X_test, y_test)

    biases.append(bias)
    variances.append(variance)

# Plote o trade-off entre bias e variância
plt.figure(figsize=(10, 6))
plt.plot(depths, biases, label='Bias', marker='o')
plt.plot(depths, variances, label='Variance', marker='o')
plt.xlabel('Profundidade da Árvore de Decisão')
plt.ylabel('Erro')
plt.title('Trade-off entre Bias e Variância')
plt.legend()
plt.grid(True)
plt.show()
