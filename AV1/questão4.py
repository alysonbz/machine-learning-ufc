# PACKAGES -------------------------------------------------------------------------------------------------------------

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# Dados ----------------------------------------------------------------------------------------------------------------
df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/Skyserver.csv')


# Encoding -------------------------------------------------------------------------------------------------------------
encoder = preprocessing.LabelEncoder()
df['class'] = encoder.fit_transform(df["class"])


# Retirando NAs --------------------------------------------------------------------------------------------------------
df.dropna(inplace=True)

# Separando alvo e preditores ------------------------------------------------------------------------------------------
X = df.drop('class', axis=1).values
y = df['class'].values

# Padronizando ---------------------------------------------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separando conjuntos de treinamento e teste ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=12)

# Inicializando os classificadores -------------------------------------------------------------------------------------
classifiers = [
    ('Árvore de Decisão', DecisionTreeClassifier(random_state=12)),
    ('Random Forest', RandomForestClassifier(random_state=12)),
    ('AdaBoost', AdaBoostClassifier(random_state=12)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=12)),
    ('Stochastic Gradient Boosting (SGB)', SGDClassifier(random_state=12))
]

# Treinando os modelos -------------------------------------------------------------------------------------------------
results = []
for clf_name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((clf_name, accuracy))

# Ordenando os resultados de forma descendente -------------------------------------------------------------------------
results.sort(key=lambda x: x[1], reverse=True)

print("Comparação de Desempenho dos Classificadores:")
for clf_name, accuracy in results:
    print(f'{clf_name}: {accuracy:.3f}')