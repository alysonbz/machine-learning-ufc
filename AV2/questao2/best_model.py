from sklearn.metrics import accuracy_score

def get_best_model(models, X_train, y_train, X_test, y_test):
    best_accuracy = 0.0
    best_model = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model
