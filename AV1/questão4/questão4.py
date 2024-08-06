
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("adult_1.1.csv")

# variáveis
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = {}

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Stochastic GB': HistGradientBoostingClassifier(random_state=42)
}

# Treinamento e avaliação dos modelos
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# resultados
df_results = pd.DataFrame(results).T
print(df_results.sort_values(by='F1 Score', ascending=False))

# Gráfico
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Comparação de Desempenho dos Modelos de Classificação', fontsize=16)
ax.set_xlabel('Modelos', fontsize=14)
ax.set_ylabel('Métricas', fontsize=14)
width = 0.2
models = df_results.index
x = range(len(models))
ax.bar(x, df_results['Accuracy'], width, label='Acurácia', color='b')
ax.bar([p + width for p in x], df_results['Precision'], width, label='Precisão', color='r')
ax.bar([p + width*2 for p in x], df_results['Recall'], width, label='Recall', color='g')
ax.bar([p + width*3 for p in x], df_results['F1 Score'], width, label='F1 Score', color='y')
ax.set_xticks([p + width*1.5 for p in x])
ax.set_xticklabels(models)
ax.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()