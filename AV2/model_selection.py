import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_classification_report(model, X_test, y_test):
    """
    Função auxiliar para gerar o classification report e a matriz de confusão para um modelo.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix

def Best_Model(X_train, X_test, y_train, y_test):
    """
    Função que realiza GridSearch para diferentes modelos de classificação e retorna o melhor modelo com base na acurácia.
    """
    # Definir os modelos
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
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 10, 20],
            'criterion': ['gini', 'entropy']
        },
        'AdaBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Loop para aplicar o GridSearch em cada modelo
    for model_name, model in models.items():
        print(f"Rodando GridSearch para {model_name}...")
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Avaliar o modelo com os dados de teste
        best_model_candidate = grid_search.best_estimator_
        y_pred = best_model_candidate.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Acurácia para {model_name}: {accuracy}")

        # Verificar se o modelo atual é o melhor
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_model_candidate
            best_model_name = model_name

    print(f"\nMelhor modelo: {best_model_name} com acurácia de {best_accuracy}")
    return best_model, best_model_name
