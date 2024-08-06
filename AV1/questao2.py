'''Faça o download do dataset e realize os pré-processamentos adequados. Selecione as colunas que você acredita ser adequdada de
analisar, remova dados desnecessários e realize uma predição utilizando árvore de decisão. Mostre números e formas adequadas de
avaliar o desempenho do classificador. Mostre, inclusive, curvas que auxiliam na análise de desempenho.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
#import scikitplot


data = pd.read_csv('weather_classification_data.csv')
#print(data.to_string())

print(data.head())
print(data.info())
print(data.isnull().sum())

colunas = {}
for column in data.select_dtypes(include=['object']).columns:
    colunas[column] = LabelEncoder()
    data[column] = colunas[column].fit_transform(data[column])


X = data.drop('Weather Type', axis=1)
y = data['Weather Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf =DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'Relatório de Classificação:\n {classification_report(y_test, y_pred)}')
print(f'Matriz de Confusão:\n {confusion_matrix(y_test, y_pred)}')
print(f'Pontuação de Precisão: {accuracy_score(y_test, y_pred)}')

print(f'Tamanho da classe: {len(set(y))}')

y_binarizar = label_binarize(y_test, classes=range(4))
y_score = clf.predict_proba(X_test)

roc_auc = roc_auc_score(y_binarizar, y_score, average='macro', multi_class='ovr')
print(f'Pontuação ROC AUC (Multicalsses): {roc_auc}')

fpr = {}
tpr = {}

plt.figure()
for i in range(len(set(y))):
    fpr[i], tpr[i], _ = roc_curve(y_binarizar[:, i], y_score[:, i])
    plt.plot(fpr[i], tpr[i], label=f'Classe {i} (área = {roc_auc_score(y_binarizar[:, i], y_score[:, i]):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC Multiclasse')
plt.legend(loc='lower right')
plt.show()