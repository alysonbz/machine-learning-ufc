# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize


# Carregar o conjunto de dados
df = pd.read_csv('dataset.csv')



df = df.fillna('0')  # Preencher valores ausentes com 0

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
    'RandomForest': {'n_estimators': [50,70], 'max_depth': [2, 3], 'random_state': [42]},
    'AdaBoost': {'n_estimators': [50, 130], 'learning_rate': [0.05, 0.2, 1]},
    'GradientBoosting': {'n_estimators': [130, 80], 'learning_rate': [0.01, 0.1]},
    'SVC': {'C': [0.06, 0.003], 'kernel': ['rbf']}
}

# Função para realizar o GridSearch e retornar a melhor parametrização de cada modelo
def perform_gridsearch(models, param_grids, X_train, y_train):
    best_params = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
        print(f"Melhores parâmetros para {model_name}: {grid_search.best_params_} com acurácia: {grid_search.best_score_:.4f}")
    return best_params

# Executar o GridSearch para encontrar a melhor parametrização
best_params = perform_gridsearch(models, param_grids, X_train, y_train)