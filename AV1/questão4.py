'''Em um unico script faça uma implementação otimizada que compare para o seu dataset, o desempenho de
arvore de decisão, random forest, adaboost, gradientBoost e SGB. Mostre numeros de forma organizada que
seja possível interpretar a melhor forma de realizar a classificação.'''

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from questao2 import X,y
import time


X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.8, random_state=42)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

resultados = {}

for clf_name, clf in classifiers.items():
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions = clf.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average = 'micro')
    recall = recall_score(y_test, predictions, average='micro')
    f1 = f1_score(y_test, predictions, average='micro')

    resultados[clf_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Training Time': train_time,
        'Prediction Time': predict_time
    }

for clf_name, metrics in resultados.items():
    print(f'Classificação: {clf_name}')
    for metrics, value in metrics.items():
        print(f'{metrics}: {value}')
    print("\n")
