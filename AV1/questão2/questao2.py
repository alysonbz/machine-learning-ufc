import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("adult.csv")

"""print(df.info())
"""

# Substituir '?' por NaN
df.replace('?', np.nan, inplace=True)
# valores ausentes
print(df.isnull().sum())
# Transformação
df.fillna('Uninformed', inplace=True)

# Verificando Outliers
numeric_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
df_zscores = df[numeric_columns].apply(zscore)  # Aplica zscore diretamente
threshold = 4
outliers = {}
for column in numeric_columns:
    outliers[column] = df[df_zscores[column].abs() > threshold].index.tolist()
outliers_summary = {column: len(outliers[column]) for column in numeric_columns}

# Exibindo um resumo dos outliers
print("Resumo de Outliers por Coluna:")
for column, count in outliers_summary.items():
    print(f"{column}: {count} outliers")


""""Os outliers detectados em algumas das colunas numéricas do dataset podem inicialmente parecer preocupantes. No entanto, considerando que esses dados vêm de uma fonte confiável como o Census Bureau e tratam-se de informações relacionadas a pessoas, essas variações são esperadas e provavelmente são legítimas.

"""

# APLICAÇÃO
# Verificando colunas para categorizar
for column in df.columns:
    print(df[column].value_counts())

columns_categorize = ["workclass","education","marital.status","occupation", "relationship", "race",'sex',]

label_encoders = {}
for column in columns_categorize:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

columns_useful = ['age', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'hours.per.week', 'capital.gain', 'capital.loss', 'income']
df = df[columns_useful]

# Transformação dos rótulos da variável 'income' para numéricos
le_income = LabelEncoder()
df['income'] = le_income.fit_transform(df['income'])  # '<=50K' para 0 e '>50K' para 1

# Divisão em features e target
X = df.drop('income', axis=1)
y = df['income']
df.to_csv("adult_1.1.csv")
# Divisão em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo de árvore de decisão
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Previsões
y_pred = tree_model.predict(X_test)
y_proba = tree_model.predict_proba(X_test)[:, 1]

# Avaliação de desempenho
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print("Accuracy: ", accuracy)
print("Classification Report:\n", report)
print("ROC AUC: ", roc_auc)

# Verificando Balanceamento das Classes
"""print(df['income'].value_counts())
"""

# grid de hiperparâmetros
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Grid Search com validação cruzada (Melhorando acuracia)
grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Melhores parâmetros e melhor score
print("\nMelhores Parâmetros:", grid_search.best_params_)
print("Melhor Acurácia:", grid_search.best_score_)