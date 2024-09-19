import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 1. Carregando os dados
# Substitua o caminho pelo local do seu dataset
df = pd.read_csv("Skyserver.csv")

# Features
X = df.drop(columns=['objid', 'specobjid', 'class'])

# Target
y = df['class']

# Colunas categóricas
categorical_columns = ['run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']

# Colunas numéricas
numerical_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']

# 2. Pré-processamento
# Definindo o pré-processamento das colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Use sparse_output=False
])

# Combinando transformações numéricas e categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função auxiliar para rodar GridSearchCV e treinar modelos
def run_grid_search(pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Accuracy no teste: {accuracy_score(y_test, y_pred)}")
    return best_model

# 3. Modelos e ajustes de parâmetros

# 3.1 Decision Tree
print("----- Decision Tree -----")
decision_tree = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', DecisionTreeClassifier())])

param_grid_dt = {
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 10],
    'classifier__min_samples_leaf': [1, 5]
}

best_dt = run_grid_search(decision_tree, param_grid_dt)

# 3.2 Random Forest
print("----- Random Forest -----")
random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier())])

param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 10],
    'classifier__min_samples_leaf': [1, 5]
}

best_rf = run_grid_search(random_forest, param_grid_rf)

# 3.3 AdaBoost
print("----- AdaBoost -----")
adaboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(algorithm='SAMME'))
])

# Definindo o grid de parâmetros
param_grid_ab = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.01, 0.1, 1.0]
}

best_ab = run_grid_search(adaboost, param_grid_ab)

# 3.4 Gradient Boosting
print("----- Gradient Boosting -----")
gradient_boost = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', GradientBoostingClassifier())])

param_grid_gb = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

best_gb = run_grid_search(gradient_boost, param_grid_gb)

# 3.5 Stochastic Gradient Boosting (SGB)
print("----- SGB (HistGradientBoosting) -----")
sgb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', HistGradientBoostingClassifier())])

param_grid_sgb = {
    'classifier__max_iter': [100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

best_sgb = run_grid_search(sgb, param_grid_sgb)

# 3.6 Support Vector Machine (SVM)
print("----- SVM -----")
svm = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', SVC())])

param_grid_svm = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

best_svm = run_grid_search(svm, param_grid_svm)
