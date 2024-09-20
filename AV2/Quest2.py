import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o dataset
df = pd.read_csv('alzheimers_disease_data.csv')

# Exclusão de dados inúteis
df = df.drop('DoctorInCharge', axis=1)

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
    'SVC': SVC()
}

param_grids = {
    'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Função para encontrar o melhor modelo baseado na acurácia
def Best_Model(X_train, y_train):
    best_model = None
    best_score = 0
    best_model_name = ''

    for model_name, model in models.items():
        grid = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_model_name = model_name

    print(f"Melhor modelo: {best_model_name} com acurácia: {best_score:.4f}")
    return best_model

# Função auxiliar para exibir classification report e matriz de confusão para todos os modelos
def evaluate_all_models(models, param_grids, X_train, y_train, X_test, y_test):
    for model_name, model in models.items():
        # Realizar o GridSearch para cada modelo
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)

        # Obter o melhor modelo e fazer previsões
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Imprimir o classification report e a matriz de confusão
        print(f"Modelo: {model_name}")
        print("Melhores parâmetros:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
        print("-" * 50)


# Executar o GridSearch e encontrar o melhor modelo
best_model = Best_Model(X_train, y_train)


# Avaliar todos os modelos
evaluate_all_models(models, param_grids, X_train, y_train, X_test, y_test)
