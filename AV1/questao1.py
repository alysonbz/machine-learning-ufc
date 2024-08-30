import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset
data = pd.read_csv('Skyserver.csv')

# Separar as features (X) e a classe alvo (y)
X = data.drop('class', axis=1)
y = data['class']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar a árvore de decisão com o índice de Gini
clf_gini = DecisionTreeClassifier(criterion="gini")
clf_gini.fit(X_train, y_train)

# Fazer previsões
y_pred_gini = clf_gini.predict(X_test)

# Avaliar a acurácia
accuracy_gini = accuracy_score(y_test, y_pred_gini)
print("Accuracy (Gini):", accuracy_gini)

# Criar a árvore de decisão com a entropia
clf_entropy = DecisionTreeClassifier(criterion="entropy")
clf_entropy.fit(X_train, y_train)

# Fazer previsões
y_pred_entropy = clf_entropy.predict(X_test)

# Avaliar a acurácia
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
print("Accuracy (Entropy):", accuracy_entropy)

