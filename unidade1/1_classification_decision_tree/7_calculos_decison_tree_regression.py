import numpy as np
import pandas as pd
from src.utils import load_house_price_dataset
import warnings
warnings.filterwarnings('ignore')

df_house = load_house_price_dataset()


def questao1(df_house):
    count = df_house['price'].count()
    average = df_house['price'].mean()
    std_dev = df_house['price'].std()
    coeff_of_variation = (std_dev / average) * 100
    return count, average, std_dev, coeff_of_variation


def questao2(df_house):
    target_column = 'price'
    # Converter a coluna target para categórica, se não for
    df_house[target_column] = pd.cut(df_house[target_column], bins=3, labels=["Low", "Medium", "High"])

    colunas_independentes = [col for col in df_house.columns if col != target_column]

    # Inicialize um dicionário para armazenar S(T, X) para cada variável
    s_tx = {}

    # Calcule S(T, X) para cada variável
    for coluna in colunas_independentes:
        if pd.api.types.is_numeric_dtype(df_house[coluna]):  # Verifica se a coluna é numérica
            s_tx[coluna] = 0
            # Calcule S(T) para cada classe
            for classe, sub_df in df_house.groupby(target_column):
                s_c = sub_df[coluna].std()  # Desvio padrão dos exemplos na classe c
                p_c = len(sub_df) / len(df_house)  # Proporção de exemplos na classe c
                s_tx[coluna] += p_c * s_c

                # Converte o resultado em um DataFrame
    resultados_df = pd.DataFrame(list(s_tx.items()), columns=['Coluna', 'S(T, X)'])

    return resultados_df


def sdr(column, target):
    if pd.api.types.is_numeric_dtype(column):  # Verifica se a coluna é numérica
        means = column.groupby(target).mean()
        stds = column.groupby(target).std()

        mean_diff = means.diff().iloc[-1]  # Diferença de médias
        std_mean = np.sqrt(stds.pow(2).mean())  # Desvio padrão médio
        sdr_value = abs(mean_diff) / std_mean if std_mean != 0 else 0

        return sdr_value
    return np.nan


def questao3(df_house):
    target_column = 'price'
    columns = [col for col in df_house.columns if col != target_column]

    # Inicializa um dicionário para armazenar SDR para cada coluna
    sdr_values = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_house[col]):  # Verifica se a coluna é numérica
            sdr_values[col] = sdr(df_house[col], df_house[target_column])

    # Converte os resultados em um DataFrame
    sdr_df = pd.DataFrame(list(sdr_values.items()), columns=['Attribute', 'SDR'])

    # Encontra o atributo com o maior SDR
    max_sdr_id = sdr_df.loc[sdr_df['SDR'].idxmax(), 'Attribute']
    print("Atributo com maior SDR:", max_sdr_id)

    return sdr_df


# Executando as funções e exibindo os resultados
print("Questao1: ", questao1(df_house))
print("Questao2: \n", questao2(df_house))
print("Questao3: \n", questao3(df_house))
