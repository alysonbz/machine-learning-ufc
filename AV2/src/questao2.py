import pandas as pd
from questao1 import best_estimator
from AV2.utils.metrics import alpha_model
from AV2.utils.processing_df import split_and_balance


def main():
    # Carregando df
    data = "../documents/adult_1.1.csv"
    df = pd.read_csv(data)

    X_train_balanced, X_test, y_train_balanced, y_test = split_and_balance(df, 'income')

    best_model, best_score = alpha_model(best_estimator, y_train_balanced, X_train_balanced)
    print(f"\nMelhor modelo: {best_model}, com acurácia de: {best_score:.4f}")


if __name__ == "__main__":
    main()



