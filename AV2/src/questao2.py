import pandas as pd
from questao1 import best_estimator
from AV2.utils.metricas_analise import melhor_modelo
from AV2.utils.db_split import split_balance


def main():
    # Carregando o dataset já ajustado anteriormente
    data = "../documents/db_ajustado.csv"
    df = pd.read_csv(data)

    # Realizando a divisão dos dados em treino (balanceado) e teste
    X_train_balanced, X_test, y_train_balanced, y_test = split_balance(df, 'quality')

    # Exibindo o melhor modelo e sua acurácia
    best_model, best_score = melhor_modelo(best_estimator, y_train_balanced, X_train_balanced)
    print(f"\nMelhor modelo: {best_model}, com acurácia de: {best_score:.4f}")


if __name__ == "__main__":
    main()



