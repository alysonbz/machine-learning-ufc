import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregando os dados
data = pd.read_csv(r"C:\Users\laura\Downloads\winequality-red.csv")
print(data)

# Verificando valores ausentes
print(data.isnull().sum())

# Separando em atributos (X) e rótulos (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializando a árvore de decisão
clf = DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=2)

# Treinando o modelo
clf.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia :", accuracy)

# Relatório de classificação
class_report = classification_report(y_test, y_pred)
print("Relatório de classificação:\n", class_report)

# classes presentes em y_test e y_pred
unique_classes = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))

# Matriz de confusão
matriz_conf = confusion_matrix(y_test, y_pred, labels=unique_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Classes Previstas")
plt.ylabel("Classes Reais")
plt.title("Matriz de Confusão")
plt.show()

# Transformando em um problema binário para cada classe
y_test_bin = label_binarize(y_test, classes=unique_classes)
y_pred_bin = label_binarize(y_pred, classes=unique_classes)
n_classes = y_test_bin.shape[1]

# Curva ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Curva ROC por classe {unique_classes[i]} (area = {roc_auc[i]:.2f}')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo ')
plt.ylabel('Verdadeiro Positivo ')
plt.title('Curva ROC para Todas as Classes')
plt.legend(loc='lower right')
plt.show()
