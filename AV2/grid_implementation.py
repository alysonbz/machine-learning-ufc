import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

df = pd.read_csv('df_final.csv')


# Dividir os dados em variáveis independentes (X) e variável dependente (y)
X = df.drop('Violence ', axis=1)
y = df['Violence ']

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir os modelos a serem avaliados
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC()
}

# Definir o espaço de parâmetros para cada modelo
param_grids = {
    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 10, 20],
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 10, 20],
        'criterion': ['gini', 'entropy']
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 10]
    },
    'SVM': {
        'C': [1, 10],  # Reduzir os valores de C
        'kernel': ['linear'],  # Focar apenas no kernel linear para ser mais rápido
        'gamma': ['scale']  # Usar apenas 'scale' ou remover 'gamma' se estiver usando o kernel linear
}
}

# Dicionário para armazenar os melhores parâmetros e relatórios de cada modelo
best_params = {}
best_classification_reports = {}

# Dicionário para armazenar os melhores parâmetros e relatórios de cada modelo
best_params = {}
best_classification_reports = {}

# Loop para aplicar o GridSearch em cada modelo
for model_name, model in models.items():
    print(f"Rodando GridSearch para {model_name}...")
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    # Ajustar o modelo usando o GridSearch
    grid_search.fit(X_train, y_train)

    # Armazenar os melhores parâmetros
    best_params[model_name] = grid_search.best_params_



# Exibir os melhores parâmetros de cada modelo
print("Melhores parâmetros para cada modelo:")
for model_name, params in best_params.items():
    print(f"{model_name}: {params}")



