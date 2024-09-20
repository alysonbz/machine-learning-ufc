import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def run_grid_search(pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Accuracy no teste: {accuracy_score(y_test, y_pred)}")
    return best_model

from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
print("----- AdaBoost -----")
adaboost = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(algorithm='SAMME'))
])

param_grid_ab = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.01, 0.1, 1.0]
}

best_ab = run_grid_search(adaboost, param_grid_ab)
