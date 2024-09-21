# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize

# Carregar o conjunto de dados
df = pd.read_csv('dataset.csv')

df = df.fillna('0')

# Obter sintomas únicos de todas as colunas de sintomas
symptom_columns = [col for col in df.columns if col.startswith('Symptom')]
all_symptoms = df[symptom_columns].values.flatten()
unique_symptoms = pd.Series(all_symptoms).unique()

# Binarizar sintomas
df['Symptoms'] = df[symptom_columns].apply(lambda x: [symptom for symptom in x if symptom != '0'], axis=1)

mlb = MultiLabelBinarizer()
binary_symptoms = mlb.fit_transform(df['Symptoms'])

df_symptoms = pd.DataFrame(binary_symptoms, columns=mlb.classes_)
df_final = pd.concat([df['Disease'], df_symptoms], axis=1)

# Remover espaços em branco extras das colunas e rótulos
df_final.columns = df_final.columns.str.strip()
df_final['Disease'] = df_final['Disease'].str.strip()


# Separar features e labels
X = df_final.drop('Disease', axis=1).values
y = df_final['Disease'].values

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


# Definir modelos e parâmetros para GridSearch
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVC': SVC()
}

param_grids = {
    'DecisionTree': {'max_depth': [35, None, 10], 'min_samples_split': [2, 5, 8]},
    'RandomForest': {'n_estimators': [50,70], 'max_depth': [2, 3]},
    'AdaBoost': {'n_estimators': [50, 130], 'learning_rate': [0.05, 0.2, 1]},
    'GradientBoosting': {'n_estimators': [130, 80], 'learning_rate': [0.01, 0.1]},
    'SVC': {'C': [0.06, 0.003], 'kernel': ['rbf']}
}

# Função para encontrar o melhor modelo baseado na acurácia
def Best_Model(X_train, y_train):
    best_model = None
    best_score = 0
    best_model_name = ''

    for model_name, model in models.items():
        grid = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='accuracy', cv=5)
        grid.fit(X_train, y_train)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_model_name = model_name

    print(f"Melhor modelo: {best_model_name} com acurácia: {best_score:.4f}")
    return best_model

# Função auxiliar para exibir classification report e matriz de confusão para todos os modelos
def evaluate_all_models(models, param_grids, X_train, y_train, X_test, y_test):
    for model_name, model in models.items():
        # Realizar o GridSearch para cada modelo
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)

        # Obter o melhor modelo e fazer previsões
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Imprimir o classification report e a matriz de confusão
        print(f"Modelo: {model_name}")
        print("Melhores parâmetros:", grid_search.best_params_)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
        print("-" * 50)

# Executar o GridSearch e encontrar o melhor modelo
best_model = Best_Model(X_train, y_train)

# Avaliar todos os modelos
evaluate_all_models(models, param_grids, X_train, y_train, X_test, y_test)