import pandas as pd
import numpy as np

def load_tabela():

        df = pd.DataFrame({
            "Resp srl": ["Pass", "Fail", "Fail", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass",
                         "Fail", "Fail", "Fail"],
            "Target": ["y", "n", "n", "y", "n", "y", "y", "y", "n", "n", "n", "n", "n", "n", "n"],
            "Predictor": ["Maths", "Maths", "Maths", "CS", "Other", "Other", "Maths", "CS", "Maths", "CS", "CS",
                          "Maths",
                          "Other", "Other", "Maths"],
            "Predictor.1": ["NW", "W", "W", "NW", "W", "W", "NW", "NW", "W", "W", "W", "NW", "W", "NW", "W"],

            "Predictor.2": ["Student", "Working", "Working", "Student", "Student", "Student", "Student", "Student",
                            "Working", "Working", "Working", "Working", "Working", "Working", "Working"]
        })

        return df

def entropia(y):
    classes, occ = np.unique(y, return_counts=True)
    pi = occ / len(y)
    entropia = - np.sum(pi * np.log2(pi))

    return entropia


def parent(df):
    value_counts = df['Exame Result'].value_counts()
    total_samples = len(df)
    P_pass = value_counts['Pass'] / total_samples
    P_fail = value_counts['Fail'] / total_samples
    return P_fail, P_fail


def tab1():
    entropia = []
    entropias = parent(entropia)
    print("Entropia de cada coluna:")
    for colunas, valores in entropias.items():
        print(f"{colunas}: {valores}")


def tab2():
    print(None)

print(tab1())