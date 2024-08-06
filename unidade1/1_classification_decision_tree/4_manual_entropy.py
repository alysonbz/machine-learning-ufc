import pandas as pd
import numpy as np

def entropy(probs):
    # Calcula a entropia dada uma lista de probabilidades
    return -np.sum(probs * np.log2(probs))


def entropy_of_list(a_list):
    # Converte uma lista de 'Y'/'N' para uma lista de 1/0 e calcula as probabilidades
    integer_list = [1 if x == 'Y' else 0 for x in a_list]
    counts = np.bincount(integer_list)
    probs = counts / len(a_list)
    return entropy(probs)


def tab1():
    # Cria um DataFrame
    df = pd.DataFrame({
        "Resp srl": ["Pass", "Fail", "Fail", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass",
                     "Fail", "Fail", "Fail"],
        "Target": ["Y", "N", "N", "Y", "N", "Y", "Y", "Y", "N", "N", "N", "N", "N", "N", "N"],
        "Predictor": ["Maths", "Maths", "Maths", "CS", "Other", "Other", "Maths", "CS", "Maths", "CS", "CS",
                      "Maths", "Other", "Other", "Maths"],
        "Predictor.1": ["NW", "W", "W", "NW", "W", "W", "NW", "NW", "W", "W", "W", "NW", "W", "NW", "W"],
        "Predictor.2": ["Student", "Working", "Working", "Student", "Student", "Student", "Student", "Student",
                        "Working", "Working", "Working", "Working", "Working", "Working", "Working"]
    })

    # índice do DataFrame
    df.set_index(["Resp srl", "Target"], inplace=True)
    return df


df = tab1()
print(df.head(15))


def tab2():
    #  entropia total do alvo
    overall_entropy = entropy_of_list(df.index.get_level_values('Target'))
    print(f"Entropia total: {overall_entropy}")

    # entropia condicional para cada preditor
    predictors = df.columns
    for predictor in predictors:
        values = df[predictor].unique()
        predictor_entropy = 0
        for value in values:
            subset = df[df[predictor] == value]
            target_entropy = entropy_of_list(subset.index.get_level_values('Target'))
            weight = len(subset) / len(df)
            predictor_entropy += weight * target_entropy
        print(f"Entropia condicional de {predictor}: {predictor_entropy}")


tab2()
