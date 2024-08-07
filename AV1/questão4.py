import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('dados1.csv')

# Separar características e alvo
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.2)

# Inicializar classificadores
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SGB": HistGradientBoostingClassifier()  # Usar HistGradientBoostingClassifier para eficiência
}

# Função para avaliar o desempenho do classificador
def evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return accuracy, conf_matrix, class_report, roc_auc

# Avaliar e armazenar desempenho para cada classificador
results = {}
for name, clf in classifiers.items():
    accuracy, conf_matrix, class_report, roc_auc = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
    results[name] = {
        "accuracy": accuracy,
        "roc_auc": roc_auc
    }
    print(f"\nDesempenho com {name}:")
    print(f'Acurácia: {accuracy:.2f}')
    print('Matriz de Confusão:')
    print(conf_matrix)
    print('Relatório de Classificação:')
    print(classification_report(y_test, clf.predict(X_test)))

    # Curva ROC
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    # Plotar curva ROC
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'{name} (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC para {name}')
    plt.legend(loc='lower right')
    plt.show()

# Gráfico de barras comparando acurácia dos modelos
names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in names]

plt.figure(figsize=(12, 8))
bars = plt.bar(names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Adicionar rótulos de valor às barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.01, f'{yval:.2f}', fontsize=12)

# Adicionar grade
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Personalizações adicionais
plt.xlabel('Modelos', fontsize=14)
plt.ylabel('Acurácia', fontsize=14)
plt.title('Comparação de Acurácia dos Modelos', fontsize=16)
plt.ylim([0, 1])
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Mostrar o gráfico
plt.tight_layout()
plt.show()

