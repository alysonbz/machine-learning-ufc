from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from questao2 import X,y
import time


# Dividir o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Inicializar os classificadores
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
# Treinar e avaliar cada classificador
results = {}

for clf_name, clf in classifiers.items():
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    predictions = clf.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')
    f1 = f1_score(y_test, predictions, average='micro')


    results[clf_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Training Time": train_time,
        "Prediction Time": predict_time,

    }

# Mostrar os resultados
for clf_name, metrics in results.items():
    print(f"Classifier: {clf_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")