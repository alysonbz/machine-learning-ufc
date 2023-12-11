# 1_AV2.py

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


def load_data():

    data = pd.read_csv(r"C:\Users\laura\Downloads\winequality-red.csv")
    return data


def prepare_data(data):

    X = data.drop('quality', axis=1)
    y = data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def grid_search_best_params(modelo, X_train, X_test, y_train, y_test):
    # Definindo os parâmetros
    parametro_grid = {}

    # GridSearchCV com o modelo e os parâmetros
    grid_search = GridSearchCV(modelo, parametro_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # ajuste com treinamento
    grid_search.fit(X_train, y_train)

    # previsões no conjunto de teste
    y_pred = grid_search.predict(X_test)
    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)

    return modelo.__class__.__name__, grid_search.best_estimator_, accuracy


def main():
    data = load_data()

    X_train, X_test, y_train, y_test = prepare_data(data)

    # modelos
    modelos = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        SGDClassifier(),
        SVC()
    ]

    # melhor modelo para cada classificador
    best_models = []
    for model in modelos:
        best_model_name, best_model, best_accuracy = grid_search_best_params(model, X_train, X_test, y_train, y_test)
        best_models.append((best_model_name, best_model, best_accuracy))

    # melhor modelo com base na acurácia
    best_model_name, best_model, best_accuracy = max(best_models, key=lambda x: x[2])

    print(f"\nMelhor Modelo: {best_model_name}  com acurácia {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
