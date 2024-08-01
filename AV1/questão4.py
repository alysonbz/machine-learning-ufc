import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('dados1.csv')

X = df.drop('Diagnosis', axis = 1)
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
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

# Avaliar e mostrar desempenho para cada classificador
for name, clf in classifiers.items():
    accuracy, conf_matrix, class_report = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
    print(f"\nDesempenho com {name}:")
    print(f'Acurácia: {accuracy:.2f}')
    print('Matriz de Confusão:')
    print(conf_matrix)
    print('Relatório de Classificação:')
    print(class_report)

 # Curva ROC
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

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