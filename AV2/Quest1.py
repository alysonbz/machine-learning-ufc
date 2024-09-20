import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregar o dataset
df = pd.read_csv('alzheimers_disease_data.csv')


# Exclusão de dados inuteis
df = df.drop('DoctorInCharge', axis = 1)


# Separar features (X) e target (y)
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir modelos e parâmetros para GridSearch
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC()
}


param_grids = {
    'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}


# Função para realizar o GridSearch e retornar a melhor parametrização de cada modelo
def perform_gridsearch(models, param_grids, X_train, y_train):
    best_params = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
        print(f"Melhores parâmetros para {model_name}: {grid_search.best_params_} com acurácia: {grid_search.best_score_:.4f}")
    return best_params

# Executar o GridSearch para encontrar a melhor parametrização
best_params = perform_gridsearch(models, param_grids, X_train, y_train)
