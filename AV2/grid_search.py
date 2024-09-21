import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

df = pd.read_csv("/home/luissavio/PycharmProjects/machine-learning-ufc/AV2/plant_growth_data_pos.csv")

X = df.drop(['Growth_Milestone'],axis=1).values
y = df[['Growth_Milestone']].values.ravel()
# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo os parâmetros para gridsearch
param_grid_tree = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20]
}

param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 10]
}

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Modelos
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC()
}

# Parâmetros para cada modelo
param_grids = {
    'Decision Tree': param_grid_tree,
    'Random Forest': param_grid_rf,
    'AdaBoost': param_grid_adaboost,
    'Gradient Boosting': param_grid_gb,
    'SVM': param_grid_svm
}

# Função para realizar o GridSearchCV
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# Aplicando o GridSearchCV para cada modelo
best_params = {}
best_scores = {}

for model_name in models.keys():
    print(f"Realizando GridSearch para {model_name}...")
    best_param, best_score = perform_grid_search(models[model_name], param_grids[model_name], X_train, y_train)
    best_params[model_name] = best_param
    best_scores[model_name] = best_score

# Exibindo os melhores parâmetros e scores
for model_name in best_params.keys():
    print(f"Melhor parametrização para {model_name}: {best_params[model_name]}")
    print(f"Melhor acurácia para {model_name}: {best_scores[model_name]:.4f}")

