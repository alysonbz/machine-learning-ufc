import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
# Carregar o dataset
df = pd.read_csv('dataset.csv')

# Preencher valores ausentes com 0
df = df.fillna('0')

# Obter sintomas únicos de todas as colunas de sintomas
symptom_columns = [col for col in df.columns if col.startswith('Symptom')]
df['Symptoms'] = df[symptom_columns].apply(lambda x: [symptom for symptom in x if symptom != '0'], axis=1)

mlb = MultiLabelBinarizer()
binary_symptoms = mlb.fit_transform(df['Symptoms'])

df_symptoms = pd.DataFrame(binary_symptoms, columns=mlb.classes_)
df_final = pd.concat([df['Disease'], df_symptoms], axis=1)

# Remover espaços em branco extras das colunas e rótulos
df_final.columns = df_final.columns.str.strip()
df_final['Disease'] = df_final['Disease'].str.strip()

# Separar features e labels
data = df_final.drop('Disease', axis=1).values
labels = df_final['Disease'].values

# Dividir em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.3, stratify=labels)
# Definindo o espaço de parâmetros
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [10, 20, 30],
    'max_iter': [100, 200, 300],
    'l2_regularization': [0, 0.1, 1, 10]}
# Definir os classificadores com parâmetros ajustados
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=35, min_samples_split=5, min_samples_leaf=4),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=3, min_samples_split=5,
                                            min_samples_leaf=4),
    'AdaBoost': AdaBoostClassifier(random_state=2, n_estimators=130, learning_rate=0.2),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=5, learning_rate=0.2, max_depth=1,
                                                    subsample=0.8),
    'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42, learning_rate=0.4,
                                                             max_depth=2,  min_samples_leaf=100)
}

results = []

# Treinar e avaliar cada classificador
for name, clf in classifiers.items():
    print(f"Treinando e avaliando {name}...")

    # Treinar o classificador
    clf.fit(x_train, y_train)

    # Fazer previsões
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else None

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None

    results.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

# Criar um DataFrame para mostrar os resultados
results_df = pd.DataFrame(results)

# Mostrar resultados
print("\nResultados de Desempenho dos Classificadores:")
print(results_df)

# Plotar gráficos de barras comparando os resultados
results_df.set_index('Classifier').plot(kind='bar', figsize=(14, 8))
plt.title('Comparação de Desempenho dos Classificadores')
plt.ylabel('Score')
plt.show()