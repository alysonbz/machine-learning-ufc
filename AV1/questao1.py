import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Carregar dados
df = pd.read_csv('Skyserver.csv')

# Codificação da coluna alvo
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Separar as características e a variável alvo
X = df.drop('class', axis=1)
y = df['class']

# Calcular ganho de informação para cada feature
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X, y)

# Obter importância das features
importances = clf.feature_importances_
feature_names = X.columns
gini_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("Importância das Features (Gini):\n", gini_importances)

# Realizar o mesmo com 'entropy' para comparar
clf_entropy = DecisionTreeClassifier(criterion='entropy')
clf_entropy.fit(X, y)
importances_entropy = clf_entropy.feature_importances_
entropy_importances = pd.Series(importances_entropy, index=feature_names).sort_values(ascending=False)

print("Importância das Features (Entropia):\n", entropy_importances)
