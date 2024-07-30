from src.utils import load_house_price_dataset
import statistics
import numpy as np
import pandas as pd

df_house = load_house_price_dataset()

def questao1():
    count = df_house['price'].count()
    avarege = df_house['price'].mean()
    standart_deviantion = statistics.stdev(df_house['price'])
    coef_of_variantion = (standart_deviantion/avarege) * 100
    return count, avarege, standart_deviantion, coef_of_variantion


def questao2(df_house):
    numeric_columns = df_house.select_dtypes(include=np.number).columns.tolist()
    stx_values = {}
    for column in numeric_columns:
        stx_values[column] = np.sqrt(np.sum((df_house[column] - df_house[column].mean()) ** 2))
    return pd.DataFrame(stx_values.items(), columns=['Column', 'STX'])


def questao3(df_house):
    numeric_columns = df_house.select_dtypes(include=np.number).columns.tolist()
    sdr_values = {}
    for column in numeric_columns:
        sdr_values[column] = np.sqrt(np.sum((df_house[column] - df_house[column].mean()) ** 2) / len(df_house))
    sdr_df = pd.DataFrame(sdr_values.items(), columns=['Column', 'SDR'])
    max_sdr_column = sdr_df['Column'][sdr_df['SDR'].idxmax()]
    return max_sdr_column

print("Questao1: ",questao1())
print("Questao2: ",questao2(df_house))
print("Questao3: ",questao3(df_house))