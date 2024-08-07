### Bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


### Dataset(dados)
df = pd.read_csv('alzheimers_disease_data.csv')

### Analises descritivas
'''print(df.columns) # 35 colunas
print(df.isnull().sum()) #0
print(df.isna().sum()) #0
print(df.shape) #(2149, 35)
print(df.info()) #22 int64  ,12 float64 e 1 object
print(df.value_counts())
print(df.dtypes)'''

# Exclusão de dados inuteis
df = df.drop('DoctorInCharge', axis = 1)

### Colunas mais importantes do dataset
### 1- Utilizando o RandomForestClassifier para obter as variaveis mais importantes

# Separar as variáveis independentes e a variável dependente
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Treinar o modelo
model = RandomForestClassifier()
model.fit(X, y)

# Obter a importância das características
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feature_importance)


### Gráfico de importância das características PARA MELHOR VISUALIZAÇÃO PARA O PROFESSOR
# Dados de importância das características
importances = {
    "FunctionalAssessment": 0.188197,
    "ADL": 0.165753,
    "MMSE": 0.120780,
    "MemoryComplaints": 0.087510,
    "BehavioralProblems": 0.051181,
    "PatientID": 0.050858,
    "SleepQuality": 0.026770,
    "PhysicalActivity": 0.024917,
    "CholesterolHDL": 0.024623,
    "BMI": 0.024109,
    "CholesterolTriglycerides": 0.023925,
    "DietQuality": 0.023667,
    "CholesterolLDL": 0.023202,
    "CholesterolTotal": 0.023126,
    "AlcoholConsumption": 0.022919,
    "SystolicBP": 0.021473,
    "DiastolicBP": 0.021116,
    "Age": 0.020837,
    "EducationLevel": 0.008158,
    "Ethnicity": 0.006982,
    "Hypertension": 0.003816,
    "Gender": 0.003521,
    "Disorientation": 0.003304,
    "Forgetfulness": 0.003299,
    "Diabetes": 0.003271,
    "FamilyHistoryAlzheimers": 0.003243,
    "Depression": 0.003170,
    "Smoking": 0.003106,
    "DifficultyCompletingTasks": 0.002889,
    "CardiovascularDisease": 0.002863,
    "PersonalityChanges": 0.002608,
    "HeadInjury": 0.002444,
    "Confusion": 0.002365
}

# Ordenar as importâncias das características
importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

# Plotar a importância das características
plt.figure(figsize=(12, 8))
plt.barh(list(importances.keys()), list(importances.values()))
plt.xlabel('Importância')
plt.ylabel('Características')
plt.title('Importância das Características')
plt.gca().invert_yaxis()
plt.show()


### Remoção de caracteristicas menos importantes
df = df.drop(['Confusion', 'HeadInjury', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Hypertension', 'Disorientation', 'Smoking', 'FamilyHistoryAlzheimers', 'Forgetfulness', 'Gender',  'CardiovascularDisease', 'Depression', 'Diabetes'], axis = 1)

# Coluna alvo Diagnosis
### Outro modelo para variaveis mais importantes

'''from sklearn.feature_selection import SelectKBest, f_classif
# Selecionar as k melhores características
k = 5
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X, y)

# Mostrar as colunas selecionadas
selected_features = selector.get_support(indices=True)
print(X.columns[selected_features])'''

### Arovre de decisao para classificação
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc


### DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state= 1, test_size=0.2)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)



print('a accuracy_score é:\n',accuracy_score(y_test, y_pred))
print('O Confusion_matrix é:\n', confusion_matrix(y_test, y_pred))
print('O classification_report é:\n', classification_report(y_test, y_pred))
print('roc_auc:\n', roc_auc_score(y_test, y_pred))


# Curva ROC
y_pred_prob = dt.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()


df.to_csv('dados1.csv', index=False)



