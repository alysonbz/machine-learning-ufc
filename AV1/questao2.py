# PACKAGES -------------------------------------------------------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Dados ----------------------------------------------------------------------------------------------------------------

df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/Skyserver.csv')


# Encoding -------------------------------------------------------------------------------------------------------------

# print('Valores unicos em "Class" antes do encoding: ', df['class'].unique())

encoder = preprocessing.LabelEncoder()
df['class'] = encoder.fit_transform(df["class"])

# print('Valores unicos em "Class" depois do encoding: ', df['class'].unique())


# Retirando NAs --------------------------------------------------------------------------------------------------------

df.dropna(inplace=True)


# Separando alvo e preditores ------------------------------------------------------------------------------------------

X = df.drop('class', axis=1)
y = df['class']


# Padronizando ---------------------------------------------------------------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Separando conjuntos de treinamento e teste ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=12)


# Treinando do modelo --------------------------------------------------------------------------------------------------
dt = DecisionTreeClassifier(max_leaf_nodes=15, random_state=12)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Acurácia do Modelo: ', acc)


# Matriz de confusão ---------------------------------------------------------------------------------------------------

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
