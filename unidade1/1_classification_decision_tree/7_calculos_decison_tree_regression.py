import numpy as np
import pandas as pd
from src.utils import load_house_price_dataset

# Carregar o dataset
df_house = load_house_price_dataset()

# Questão 1: Calcular Count, Average, Standard Deviation e Coefficient of Variation da coluna 'price'
def questao1(df):
    count = df['price'].count()
    avg = df['price'].mean()
    std = df['price'].std()
    cv = (std / avg) * 100
    return count, avg, std, cv

# Questão 2: Calcular S(T,X) para todas as colunas, exceto 'price'
def questao2(df):
    columns = df.columns.difference(['price'])
    S_values = []

    for column in columns:
        correlation = df[column].corr(df['price'])
        S = correlation * df['price'].std()
        S_values.append({'variavel': column, 'S': S})

    S_df = pd.DataFrame(S_values)
    S_df = S_df.sort_values(by='S', ascending=False).reset_index(drop=True)
    return S_df

# Questão 3: Calcular SDR(T,X) para todas as colunas, exceto 'price', e encontrar o atributo com o maior SDR
def questao3(df):
    columns = df.columns.difference(['price'])
    SDR_values = []

    for column in columns:
        std_price = df['price'].std()
        std_column = df[column].std()
        SDR = std_price - (df[column].corr(df['price']) * std_price * (std_price / std_column))
        SDR_values.append({'variavel': column, 'SDR': SDR})

    SDR_df = pd.DataFrame(SDR_values)
    max_SDR_variable = SDR_df.loc[SDR_df['SDR'].idxmax(), 'variavel']
    return SDR_df, max_SDR_variable

# Executar as questões
count, avg, std, cv = questao1(df_house)
print("Questão 1:")
print(f"Count: {count}")
print(f"Average: {avg:.2f}")
print(f"Standard Deviation: {std:.2f}")
print(f"Coefficient of Variation: {cv:.2f}\n")

print("Questão 2:")
S_df = questao2(df_house)
print(S_df, "\n")

print("Questão 3:")
SDR_df, max_SDR_variable = questao3(df_house)
print(SDR_df)
print(f"Maior SDR: {max_SDR_variable}")
