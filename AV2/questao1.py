from AV1.questao2 import X, y
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

def get_best_params(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

if __name__ == "__main__":


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir os modelos
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Stochastic Gradient Boosting": SGDClassifier(loss='modified_huber', shuffle=True, random_state=42),
        "SVM": SVC()
    }

    params = {
        "Decision Tree": {"max_depth": [3, 5, 7]},
        "Random Forest": {"n_estimators": [50, 100, 150]},
        "AdaBoost": {"n_estimators": [50, 100, 150], "learning_rate": [0.1, 0.5, 1.0]},
        "Gradient Boosting": {"n_estimators": [50, 100, 150], "learning_rate": [0.1, 0.5, 1.0]},
        "Stochastic Gradient Boosting": {"max_iter": [1000, 1500, 2000]},
        "SVM": {"C": [1, 10, 100], "gamma": [0.1, 1, 10]}
    }


    best_params = {}
    for model_name, model in models.items():
        best_params[model_name] = get_best_params(model, params[model_name], X_train, y_train)
        print(f"Melhores parâmetros para {model_name}: {best_params[model_name]}")


    trained_models = {}
    for model_name, model in models.items():
        model.set_params(**best_params[model_name])
        model.fit(X_train, y_train)
        trained_models[model_name] = model


    for model_name, model in trained_models.items():
        accuracy = model.score(X_test, y_test)
        print(f"Acurácia do modelo {model_name} no conjunto de teste: {accuracy:.4f}")