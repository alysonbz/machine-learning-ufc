# run_grid_search.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

def Best_Model(X_train, y_train):
    models = {
        'Decision Tree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 10, 20, 30]}),
        'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 150], 'classifier__max_depth': [None, 10, 20, 30]}),
        'AdaBoost': (AdaBoostClassifier(), {'classifier__n_estimators': [50, 100, 150], 'classifier__learning_rate': [0.01, 0.1, 1.0]}),
        'Gradient Boosting': (
            GradientBoostingClassifier(), {'classifier__n_estimators': [50, 100, 150], 'classifier__learning_rate': [0.01, 0.1, 1.0]}),
        'SVM': (SVC(), {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto'], 'classifier__kernel': ['linear', 'rbf']})
    }

    best_model = None
    best_accuracy = 0.0

    for name, (model, params) in models.items():
        # Definir transformações específicas para colunas categóricas
        categorical_features = ['workclass', 'occupation']
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combinar transformações
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
            ])

        # Adicionar o modelo e o pré-processador ao pipeline
        model_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Realizar grid search com validação cruzada
        grid_search = GridSearchCV(model_pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Obter a acurácia e verificar se é a melhor até agora
        accuracy = grid_search.best_score_
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid_search.best_estimator_

    return best_model

def get_classification_report_and_confusion_matrix(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[name] = {
            'Accuracy': accuracy,
            'Classification Report': classification_rep,
            'Confusion Matrix': conf_matrix
        }

    return results

def main():
    # Carregar conjunto de dados
    df = pd.read_csv('C:\\Users\\anime\\PycharmProjects\\machine-learning-ufc\\adult_novo.csv')

    # Separar features (X) e target (y)
    X = df.drop('income', axis=1)
    y = df['income']

    # Dividir conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encontrar o melhor modelo com base na acurácia
    best_model = Best_Model(X_train, y_train)

    # Armazenar os modelos para análises adicionais
    models = {
        'Decision Tree': DecisionTreeClassifier(),  # Não usar pré-processamento aqui
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC()
    }

    # Pré-processamento específico para colunas categóricas
    categorical_features = ['workclass', 'occupation']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])

    # Aplicar pré-processamento a todos os modelos
    for name, model in models.items():
        model_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        model_pipe.fit(X_train, y_train)
        models[name] = model_pipe

    # Obter resultados detalhados (acurácia, classification report, e matriz de confusão) para todos os modelos
    results = get_classification_report_and_confusion_matrix(models, X_test, y_test)

    # Exibir resultados
    print(f"Best Model: {best_model.named_steps['classifier']}")
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {result['Accuracy']:.4f}")
        print("Classification Report:")
        print(result['Classification Report'])
        print("Confusion Matrix:")
        print(result['Confusion Matrix'])

if __name__ == "__main__":
    main()

