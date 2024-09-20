from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


def print_metrics(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def melhor_modelo(models, y_test, X_test):
    best_model = None
    best_score = 0

    for model_name, model in models.items():
        print(f"\n### Relatório para {model_name} ###")
        y_pred = model.predict(X_test)
        print_metrics(y_test, y_pred)

        score = accuracy_score(y_test, y_pred)
        print(f"Acurácia para {model_name}: {score:.4f}")

        if score > best_score:
            best_model = model
            best_score = score

    return best_model, best_score
