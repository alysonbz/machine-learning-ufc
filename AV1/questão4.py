import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Conjunto de dados
data = pd.read_csv(r"C:\Users\laura\Downloads\winequality-red.csv")

X = data.drop("quality", axis=1)
y = data["quality"]

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializando os classificadores
classificadores = {
    "Árvore de Decisão": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Stochastic Gradient Boosting": SGDClassifier()
}

# Resultados
resultados = {"Classificador": [], "Acurácia": []}

# Treinando cada classificador
for clf_name, clf in classificadores.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados["Classificador"].append(clf_name)
    resultados["Acurácia"].append(accuracy)

# DataFrame para visualizar os resultados
resultados_df = pd.DataFrame(resultados)

# Acurácia dos classificadores
resultados_df = resultados_df.sort_values(by="Acurácia", ascending=False)
print(resultados_df)

