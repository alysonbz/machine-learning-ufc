from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from questao1 import grid_search_models, prepare_data, return_metrics

def return_metrics(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def alpha_model(models, y_test, X_test):
    best_model = None
    best_score = 0

    for model_name, model in models.items():
        print(f"\n### Relatório para {model_name} ###")
        y_pred = model.predict(X_test)
        return_metrics(y_test, y_pred)

        score = accuracy_score(y_test, y_pred)
        print(f"Acurácia para {model_name}: {score:.4f}")

        if score > best_score:
            best_model = model
            best_score = score

    return best_model, best_score


def main():
    # Carregando df
    data = pd.read_csv(
        'C:/Users/Luciana/OneDrive/Documentos/aprendizado_maquina/machine-learning-ufc/AV1/weather_classification_data.csv')

    # Preparando os dados
    X_train_balanced, X_test, y_train_balanced, y_test = prepare_data(data)

    # Executando o GridSearch para encontrar os melhores modelos
    best_model, best_params, best_score = grid_search_models(X_train_balanced, y_train_balanced)

    # Usando o melhor modelo encontrado para prever no conjunto de teste
    print("\n### Relatório para o melhor modelo encontrado ###")
    y_pred = best_model.predict(X_test)
    return_metrics(y_test, y_pred)
    print(f"\nMelhor modelo: {best_model}, com acurácia de: {best_score:.4f}")


if __name__ == "__main__":
    main()
