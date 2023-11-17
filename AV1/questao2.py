# ------------------- BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("dataset.csv")
df = shuffle(df,random_state=56)

print(df.head())
print("------------------------------------------------------------\n")
print(df.info())
print("-------------------------------------------------------------\n")
print(df.describe())

print("------------------ PRÉ-PROCESSAMENTO")


def remove_space_between_word(dataset):
    for col in dataset.columns:
        for i in range(len(dataset[col])):
            if (type(dataset[col][i]) == str ):
                dataset[col][i] = dataset[col][i].strip()
                dataset[col][i] = dataset[col][i].replace(" ", "_")
    return dataset


df = remove_space_between_word(df)
print(df.head())
print("-------------------------------------------------------------\n")
print("CHECANDO NAS")
null = df.isnull().sum().to_frame(name='count')
print(null)
print("-------------------------------------------------------------\n")
df = df.fillna(0)
print(df.head())

print("DATASET SOBRE A SEVERIDADE DOS SÍNTOMAS")

df_1 = pd.read_csv("Symptom-severity.csv")
print(df_1.head())
print("-----------------------------------------------------------")
symptoms = df_1.iloc[:,0].unique()

print(symptoms)


vals = df.values

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df_1[df_1['Symptom'] == symptoms[i]]['weight'].values[0]

df = pd.DataFrame(vals, columns= df.columns)
print(df.head())

df = df.replace("foul_smell_of_urine", 0)
df = df.replace("dischromic__patches", 0)
df = df.replace("spotting__urination", 0)

print(df.columns)
print("--------------------------------------------------")
print("DECISION TREE")

# Separar as features (X) e o target (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("--------------------------------------------------")

# Inicializar e treinar o modelo de árvore de decisão
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
predictions = tree.predict(X_test)

# Calcular e exibir a acurácia
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia: {accuracy}')

# Exibir a matriz de confusão
conf_matrix = confusion_matrix(y_test, predictions)
print('Matriz de Confusão:')
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=tree.classes_, yticklabels=tree.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Exibir o relatório de classificação
class_report = classification_report(y_test, predictions)
print('Relatório de Classificação:')
print(class_report)

