import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
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

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def grid_search_best_params(modelo, parametro_grid, X_train, y_train):
    grid_search = GridSearchCV(modelo, parametro_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_parametros = grid_search.best_params_
    return best_parametros

def evaluate_model(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Acurácia: {accuracy}")
    print("Relatório de Classificação:\n", report)

def main():
    data = load_data()

    X_train, X_test, y_train, y_test = prepare_data(data)

    # modelos e parâmetros para grid search
    modelos = {
        'Decision Tree': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30]}),
        'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}),
        'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}),
        'Gradient Boosting': (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}),
        'Stochastic Gradient Boosting': (SGDClassifier(), {'max_iter': [100, 200, 300], 'alpha': [0.0001, 0.001, 0.01]}),
        'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
    }

    # grid search
    best_parametros = {}
    best_modelos = {}
    for modelo_nome, (modelo, parametros_grid) in modelos.items():
        print(f"\nGrid Search para {modelo_nome}...")
        best_parametros[modelo_nome] = grid_search_best_params(modelo, parametros_grid, X_train, y_train)
        print(f"Best Parametros: {best_parametros[modelo_nome]}")

        # Treinando o modelo com os melhores parâmetros
        modelo.set_params(**best_parametros[modelo_nome])
        modelo.fit(X_train, y_train)
        best_modelos[modelo_nome] = modelo

        # Avaliando o modelo no conjunto de teste
        print(f"\nEvaluation for {modelo_nome} on Test Set:")
        evaluate_model(modelo, X_test, y_test)

    print("\nBest Parameters:")
    for modelo_nome, parametros in best_parametros.items():
        print(f"{modelo_nome}: {parametros}")

if __name__ == "__main__":
    main()
