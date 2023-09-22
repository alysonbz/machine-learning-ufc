
from src.utils import load_house_price_dataset
import pandas as pd
import numpy as np
df = load_house_price_dataset()
""""
1) Na função ``questão1`` realize os cáculos de count, average, Standard Deviation e Coeff. of Variation da coluna price do dataset. Retorne os valores.
2) na função ``questao2`` realize os cáulos de S(T,X) para todas as colunas do dataset, com exceção da coluna target. Armazene as respostas em um dataframe e retorne.
3) na função ``questão3`` realize o calculo de SDR(T,X) para todas as colunas do dataset, exceção da coluna target. Amazene a resposta em um dataframe e retorne o id do atributo que possui o maior SDR.
"""


# 1) Função para calcular count, média, desvio padrão e coeficiente de variação da coluna 'price'
def questao1(df):
    count = df['price'].count()
    average = df['price'].mean()
    std_deviation = df['price'].std()
    coeff_of_variation = (std_deviation / average) * 100  # Em percentagem

    return count, average, std_deviation, coeff_of_variation

# 2) Função para calcular S(T,X) para todas as colunas, exceto a coluna alvo
def questao2(df):
    columns = df.filter(like="^.*", exclude="price")

    results = {}

    for col in columns:
        dt = df[col].value_counts().reset_index()
        std = df.groupby(col)['price'].std()
    return pd.Series(results)

# 3) Função para calcular SDR(T,X) para todas as colunas, exceto a coluna alvo
def questao3(df):


print()