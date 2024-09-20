import warnings
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

warnings.filterwarnings("ignore")

data = pd.read_csv('C:/Users/Luciana/OneDrive/Documentos/aprendizado_maquina/machine-learning-ufc/AV1/weather_classification_data.csv')
# conversão de variaveis categoricas para numericas
def prepare_data(data, target_column='Weather Type'):
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sm = SMOTE(random_state=1)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

def new_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('classifier', model)])

def return_metrics(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def grid_search_models(X_train, y_train):
    param_grid = {
        'decision_tree': {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [5, 10, 15, 20],
            'classifier__min_samples_split': [10, 15, 20, 25, 30]
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, 20, 30],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__min_samples_split': [10, 15, 20, 25, 30]
        },
        'adaboost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1],
        },
        'gradient_boost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, 20, 30],
            'classifier__learning_rate': [0.01, 0.1]
        },
        'svc': {
            'classifier__kernel': ['linear', 'sigmoid', 'rbf'],
            'classifier__C': [0.01, 1, 10],
            'classifier__max_iter': [100, 1000, 2000]
        },
    }

    models = {
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'adaboost': AdaBoostClassifier(),
        'gradient_boost': GradientBoostingClassifier(),
        'svc': SVC()
    }

    best_model = None
    best_score = 0
    best_params = None

    for model_name, model in models.items():
        pipeline = new_pipeline(model)
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        print(f'Melhores parâmetros para {model_name}: {grid_search.best_params_}')

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
    return best_model, best_params, best_score


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(data)
    best_model, best_params, best_score = grid_search_models(X_train, y_train)
    print(f"\nMelhor modelo: {best_model}")
    print(f"Melhores parâmetros: {best_params}")
    print(f"Melhor acurácia: {best_score:.4f}")

    # Avaliação no conjunto de teste
    y_pred = best_model.predict(X_test)
    return_metrics(y_test, y_pred)