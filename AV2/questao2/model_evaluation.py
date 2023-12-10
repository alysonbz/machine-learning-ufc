from sklearn.metrics import classification_report, confusion_matrix

def evaluate_models(models, X_train, y_train, X_test, y_test):
    evaluation_results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)

        evaluation_results[model_name] = {
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat
        }

    return evaluation_results
