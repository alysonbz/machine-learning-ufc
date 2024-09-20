import pandas as pd
from questao1 import best_estimator
from AV2.utils.calculo_metricas import Best_Model
from AV2.utils.dataset_split import split


def questao2():
    # Carregando df
    data = "../documents/data.csv"
    df = pd.read_csv(data)

    X_train, X_test, y_train, y_test = split(df, 'fail')

    best_model, best_score = Best_Model(best_estimator, y_train, X_train)
    print(f"\nMelhor modelo: {best_model}, com acurácia de: {best_score:.4f}")


if __name__ == "__main__":
    questao2()



