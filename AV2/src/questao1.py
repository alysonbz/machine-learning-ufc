from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import warnings
from AV2.utils.dataset_split import split
from AV2.utils.pipeline import meu_pipeline
warnings.filterwarnings('ignore')
#Carregando o dataset
dados = "../documents/data.csv"
df = pd.read_csv(dados)

#Divisão em treino e teste
X_train, X_test, y_train, y_test = split(df, 'fail')

#Definindo os parametros do GridSearch
parametros_grid = {
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
        'classifier__max_iter': [100, 1000, 2000, 10000]
    },
    'HistGB': {
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_iter': [100, 1000, 2000],
        'classifier__max_depth': [5, 10, 20, 30]
    }

}
# Instanciando os modelos
models = {
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'adaboost': AdaBoostClassifier(),
    'gradient_boost': GradientBoostingClassifier(),
    'svc': SVC(),
    'HistGB': HistGradientBoostingClassifier()
}

best_estimator = {}

#Executando o pipeline e param_grid
for model_name, model in models.items():
    pipeline = meu_pipeline(model)
    parametros_grids = parametros_grid[model_name]
    grid_search = GridSearchCV(pipeline, parametros_grids, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Salvando melhor estimador
    best_estimator[model_name] = grid_search.best_estimator_
    print(f'Melhores paramêtros para {model_name}: {grid_search.best_params_}')

