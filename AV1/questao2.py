import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Carregando os dados
df = pd.read_csv('winequality-red.csv')

# Analisando se há existência de valores nulos
print(df.head())
print(df.isnull().sum())

# Analisando a distribuição das classes
print("Distribuição das classes:\n", df["quality"].value_counts())

# Transformando a variável de saída 'quality' em duas classes: 0 (qualidade ruim) e 1 (qualidade boa)
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Salvando a base de dados modificada
df.to_csv('db_ajustado.csv', index=False)

# Analisando a nova distribuição das classes
print("Nova distribuição das classes:\n", df["quality"].value_counts())

# Selecionando as colunas mais adequadas
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
y = df['quality'].values

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Balanceando as classes usando SMOTE
smote = SMOTE(random_state=1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Analisando a distribuição das classes após o balanceamento
print("Distribuição das classes após o balanceamento:\n", pd.Series(y_train_bal).value_counts())

# Instanciando o classificador DecisionTreeClassifier com profundidade máxima de 6
dt = DecisionTreeClassifier(max_depth=6)

# Ajustando o classificador aos dados de treinamento balanceados
dt.fit(X_train_bal, y_train_bal)

# Prevendo os rótulos do conjunto de teste
y_pred = dt.predict(X_test)

# Imprimindo os rótulos previstos
print("Rótulos previstos:", y_pred)

# Calculando a acurácia do conjunto de teste
acc = accuracy_score(y_test, y_pred)
print("Acurácia do conjunto de teste: {:.2f}".format(acc))

# Verificando Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:\n", cm)

# Verificando Relatório de classificação
cr = classification_report(y_test, y_pred)
print("Relatório de classificação:\n", cr)

# Aplicando Curva ROC
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
