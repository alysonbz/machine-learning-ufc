# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from src.utils import load_breast_cancer_dataset
from sklearn.model_selection import train_test_split

# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]].values
y = df_breast[['diagnosis']].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Instancie as árvores de decisão com 2 possíveis critérios: entropy e gini
dt_entropy = DecisionTreeClassifier(max_depth=20,criterion="entropy", random_state=42)
dt_gini =DecisionTreeClassifier(max_depth=20,criterion="gini", random_state=42)

# Fit os objetos dt_gini e dt_entropy
dt_entropy.fit(X_train, y_train)
dt_gini.fit(X_train, y_train)


# Use dt_entropy e dt_gini para realizar predições no conjunto de teste
y_pred_entropy= dt_entropy.predict(X_test)
y_pred_gini = dt_gini.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

accuracy_gini = accuracy_score(y_test, y_pred_gini)

# Print accuracy_entropy
print(f'Acuráia de Entropia: {accuracy_entropy}')

# Print accuracy_gini
print(f'Acuráia de Gini: {accuracy_gini}')
