from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
import pandas as pd
import warnings
from AV2.utils.workflow import new_pipeline
from AV2.utils.grid import param_grid
from AV2.utils.processing_df import split_and_balance

warnings.filterwarnings("ignore")

# Carregando df
data = "../documents/adult_1.1.csv"
df = pd.read_csv(data)

X_train_balanced, X_test, y_train_balanced, y_test = split_and_balance(df, 'income')

models = {
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'adaboost': AdaBoostClassifier(),
    'gradient_boost': GradientBoostingClassifier(),
    'svc': SVC(),
    'HistGradientBoosting': HistGradientBoostingClassifier()
}

best_estimator = {}
for model_name, model in models.items():
    pipeline = new_pipeline(model)
    param_grids = param_grid[model_name]
    grid_search = GridSearchCV(pipeline, param_grids, cv=5, scoring='accuracy')
    grid_search.fit(X_train_balanced, y_train_balanced)

    # Salvando melhor estimador
    best_estimator[model_name] = grid_search.best_estimator_
    print(f'Melhores paramêtros para {model_name}: {grid_search.best_params_}')
