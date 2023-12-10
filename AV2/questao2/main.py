from best_model import get_best_model
from model_evaluation import evaluate_models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from AV1.questao2 import X, y


def main():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Stochastic Gradient Boosting": SGDClassifier(loss='modified_huber', shuffle=True, random_state=42),
        "SVM": SVC()
    }


    best_model = get_best_model(models,  X_train, y_train,X_test, y_test)

    evaluation_results = evaluate_models(models, X_train, y_train, X_test, y_test)

    print("Melhor modelo encontrado:")
    print(best_model)

    print("\nResultados da avaliação de todos os modelos:")
    for model_name, results in evaluation_results.items():
        print(f"Modelo: {model_name}")
        print("Classification Report:")
        print(results['classification_report'])
        print("Confusion Matrix:")
        print(results['confusion_matrix'])
        print("--------------------")

if __name__ == "__main__":
    main()
