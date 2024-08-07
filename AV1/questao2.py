import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize

# Carregar o dataset
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
data = df_final.drop('Disease', axis=1).values
labels = df_final['Disease'].values

# Dividir em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.3)

# Random Forest
print("\nRandom Forest")
rf = RandomForestClassifier(random_state=1, max_depth=4)
rf.fit(x_train, y_train)

# Importância das features
feature_importances = rf.feature_importances_
features = df_final.drop('Disease', axis=1).columns

# Criar um dataframe para as importâncias das features
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Ordenar por importância
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotar as importâncias
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importância das Features')
plt.show()

# Mostrar as colunas mais relevantes
print("\nColunas mais relevantes:")
print(importance_df.head(10))

y_pred_rf = rf.predict(x_test)
print('Accuracy% =', accuracy_score(y_test, y_pred_rf) * 100)
print('O classification_report é:\n', classification_report(y_test, y_pred_rf))

# Cálculo de ROC AUC para multi-classe
y_test_bin = label_binarize(y_test, classes=rf.classes_)
y_pred_rf_bin = label_binarize(y_pred_rf, classes=rf.classes_)
roc_auc = roc_auc_score(y_test_bin, y_pred_rf_bin, average='macro')
print('roc_auc:\n', roc_auc)


cm = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Dividir as classes em dois grupos
num_classes = len(rf.classes_)
half_num_classes = num_classes // 2

# Função para plotar a curva ROC para um grupo de classes
def plot_roc_curve(classes, x_test, y_test, y_proba, title):
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(classes):
        y_true_bin = (y_test == class_name).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
        plt.plot(fpr, tpr, label=f'ROC curve of class {class_name} (area = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

# Prever probabilidades para todas as classes
y_proba = rf.predict_proba(x_test)

# Plotar a curva ROC para as primeiras 10 classes
first_10_classes = rf.classes_[:10]
plot_roc_curve(first_10_classes, x_test, y_test, y_proba, 'ROC Curves for First 10 Classes')

# Plotar a curva ROC para as próximas 10 classes
next_10_classes = rf.classes_[10:20]
plot_roc_curve(next_10_classes, x_test, y_test, y_proba, 'ROC Curves for Next 10 Classes')