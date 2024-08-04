import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Carregar dados
df = pd.read_csv('Skyserver.csv')

# Pré-processamento
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Separar características e variável alvo
X = df.drop('class', axis=1)
y = df['class']

# Binarizar as classes para o cálculo da curva ROC
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# Treinamento do modelo
clf = OneVsRestClassifier(DecisionTreeClassifier(criterion='gini'))
clf.fit(X_train, y_train)

# Previsões
y_pred = clf.predict(X_test)

# Avaliação do modelo
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Curvas ROC
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test[:, i], clf.predict_proba(X_test)[:, i])
    plt.plot(fpr, tpr, label=f'Classe {i} (área = {auc(fpr, tpr):.2f})')

plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC para cada classe')
plt.legend(loc='lower right')
plt.show()
