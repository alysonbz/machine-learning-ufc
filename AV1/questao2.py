import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Carregar os dados
df = pd.read_csv("data.csv")

# Verificar a existência de nulos
print(df.isna().sum())

# Verificar a distribuição de classes
print("Distribuição de classes:\n", df["fail"].value_counts())

# Selecionar as variáveis adequadas
X = df[["footfall","tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"]]
y = df["fail"]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Aplicar SMOTE para balancear as classes no conjunto de treinamento
smote = SMOTE(random_state=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verificar a nova distribuição de classes no conjunto de treinamento balanceado
print("Distribuição de classes após balanceamento:\n", pd.Series(y_train_balanced).value_counts())

# Instanciar o classificador DecisionTreeClassifier com profundidade máxima de 6
dt = DecisionTreeClassifier(max_depth=6)

# Ajustar o classificador aos dados de treinamento balanceados
dt.fit(X_train_balanced, y_train_balanced)

# Prever os rótulos do conjunto de teste
y_pred = dt.predict(X_test)

# Imprimir os rótulos previstos
print("Rótulos previstos:", y_pred)

# Calcular a acurácia do conjunto de teste
acc = accuracy_score(y_test, y_pred)
print("Acurácia do conjunto de teste: {:.2f}".format(acc))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:\n", cm)

# Relatório de classificação
cr = classification_report(y_test, y_pred)
print("Relatório de classificação:\n", cr)

# Curva ROC
y_prob = dt.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Decision Tree (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
