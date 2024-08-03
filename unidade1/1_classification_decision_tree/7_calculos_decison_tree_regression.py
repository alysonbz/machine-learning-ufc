#from src.utils import load_house_price_dataset
import pandas as pd
import numpy as np

df_house = pd.read_csv("../dataset/Housing.csv")

def questao1():
    count = df_house['price'].count()
    average = df_house['price'].mean()
    std_deviation = df_house['price'].std()
    coefficient_of_variation = std_deviation / average
    return count, average, std_deviation, coefficient_of_variation
def questao2():
    numerical_cols = df_house.select_dtypes(include=[np.number]).columns.tolist()
    result_dict = {}
    for col in numerical_cols:
        try:
            s_tx = df_house['price'].cov(df_house[col]) / df_house[col].var()
            result_dict[col] = s_tx
        except:
            pass
    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['S(T,X)'])
    return result_df
def questao3():
    numerical_cols = df_house.select_dtypes(include=[np.number]).columns.tolist()
    max_sdr_col = None
    max_sdr_value = -np.inf
    for col in numerical_cols:
        try:
            sdr = abs(df_house['price'].corr(df_house[col])) * (df_house[col].std() / df_house['price'].std())
            if sdr > max_sdr_value:
                max_sdr_value = sdr
                max_sdr_col = col
        except:
            pass
    return max_sdr_col

print("Questao1: ",questao1())
print("Questao2: ",questao2())
print("Questao3: ",questao3())