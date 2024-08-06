
from src.utils import load_house_price_dataset
import pandas as pd
import numpy as np
df = load_house_price_dataset()

def sdr(column, target):
    if pd.api.types.is_numeric_dtype(column):  # Verifica se a coluna é numérica
        means = column.groupby(target).mean()
        stds = column.groupby(target).std()

        mean_diff = means.diff().iloc[-1]  # Diferença de médias
        std_mean = np.sqrt(stds.pow(2).mean())  # Desvio padrão médio
        sdr_value = abs(mean_diff) / std_mean if std_mean != 0 else 0

        return sdr_value
    return np.nan

def questao1():
    count = df['price'].count()
    average = df['price'].mean()
    std_deviation = df['price'].std()
    coefficient_of_variation = std_deviation / average
    return count, average, std_deviation, coefficient_of_variation

def questao2():
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result_dict = {}
    for col in numerical_cols:
        try:
            s_tx = df['price'].cov(df[col]) / df[col].var()
            result_dict[col] = s_tx
        except:
            pass
    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['S(T,X)'])
    return result_df


def questao3():
    target_column = 'price'
    columns = [col for col in df.columns if col != target_column]

    # dicionário para armazenar SDR para cada coluna
    sdr_values = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Verifica se a coluna é numérica
            sdr_values[col] = sdr(df[col], df[target_column])

    # Converte os resultados em um DataFrame
    sdr_df = pd.DataFrame(list(sdr_values.items()), columns=['Attribute', 'SDR'])

    # atributo com o maior SDR
    max_sdr_id = sdr_df.loc[sdr_df['SDR'].idxmax(), 'Attribute']
    print("\nAtributo com maior SDR:", max_sdr_id)

    return sdr_df



print("Questao1: ",questao1())
print("\nQuestao2: ",questao2())
print("Questao3: ",questao3())