from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar os dados
data = pd.read_csv("Skyserver.csv")

# Pré-processamento dos dados
X = data.drop('class', axis=1)
y = data['class']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir modelos
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Avaliar modelos
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Exibir resultados
print("Resultados de Precisão dos Modelos:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2f}")
