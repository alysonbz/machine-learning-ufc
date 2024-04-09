
from src.utils import load_house_price_dataset
import pandas as pd
import numpy as np
df = load_house_price_dataset()


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
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    max_sdr_col = None
    max_sdr_value = -np.inf
    for col in numerical_cols:
        try:
            sdr = abs(df['price'].corr(df[col])) * (df[col].std() / df['price'].std())
            if sdr > max_sdr_value:
                max_sdr_value = sdr
                max_sdr_col = col
        except:
            pass
    return max_sdr_col

print("Questao1: ",questao1())
print("\nQuestao2: ",questao2())
print("\nQuestao3: ",questao3())