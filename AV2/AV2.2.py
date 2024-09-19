import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class ModelSelector:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        self.best_accuracy = 0
        self.models = {}

    def fit(self, pipelines, param_grids):
        for name, pipeline in pipelines.items():
            param_grid = param_grids[name]
            print(f"Rodando GridSearch para {name}...")
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            best_accuracy = grid_search.best_score_

            self.models[name] = {
                'model': best_model,
                'accuracy': best_accuracy,
                'best_params': grid_search.best_params_
            }

            if best_accuracy > self.best_accuracy:
                self.best_accuracy = best_accuracy
                self.best_model = best_model

    def get_best_model(self):
        return self.best_model, self.best_accuracy

    def generate_reports(self):
        for name, info in self.models.items():
            model = info['model']
            print(f"----- Relatório do {name} -----")
            y_pred = model.predict(self.X_test)
            print(f"Acurácia: {accuracy_score(self.y_test, y_pred)}")
            print("Relatório de Classificação:")
            print(classification_report(self.y_test, y_pred))
            print("Matriz de Confusão:")
            print(confusion_matrix(self.y_test, y_pred))
            print()


# 1. Carregar os dados
df = pd.read_csv("Skyserver.csv")

# Features e target
X = df.drop(columns=['objid', 'specobjid', 'class'])
y = df['class']

# Definir colunas
categorical_columns = ['run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']
numerical_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']

# 2. Pré-processamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir pipelines e grids
pipelines = {
    'DecisionTree': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', DecisionTreeClassifier())]),
    'RandomForest': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', RandomForestClassifier())]),
    'AdaBoost': Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', AdaBoostClassifier(algorithm='SAMME'))]),
    'GradientBoosting': Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', GradientBoostingClassifier())]),
    'HistGradientBoosting': Pipeline(steps=[('preprocessor', preprocessor),
                                            ('classifier', HistGradientBoostingClassifier())]),
    'SVM': Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SVC())])
}

param_grids = {
    'DecisionTree': {
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 10],
        'classifier__min_samples_leaf': [1, 5]
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 10],
        'classifier__min_samples_leaf': [1, 5]
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1, 1.0]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'HistGradientBoosting': {
        'classifier__max_iter': [100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }
}

# Criar o seletor de modelos e ajustar
model_selector = ModelSelector(X_train, y_train, X_test, y_test)
model_selector.fit(pipelines, param_grids)

# Obter o melhor modelo e precisão
best_model, best_accuracy = model_selector.get_best_model()
print(f"Melhor Modelo com Acurácia: {best_accuracy}")

# Gerar relatórios
model_selector.generate_reports()
