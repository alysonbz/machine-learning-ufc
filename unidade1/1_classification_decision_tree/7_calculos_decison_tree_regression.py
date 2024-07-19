import pandas as pd
import numpy as np
from src.utils import load_house_price_dataset

df_house = load_house_price_dataset()


def questao1():
    count = df_house['price'].count()
    average = df_house['price'].mean()
    std_dev = df_house['price'].std()
    coeff_var = std_dev / average

    return {
        "count": count,
        "average": average,
        "std_dev": std_dev,
        "coeff_var": coeff_var
    }


def calculate_sx(df, feature, target):
    # Calcula S(T, X) que é a soma das variâncias ponderadas das partições feitas pela coluna feature
    total_var = 0
    unique_values = df[feature].unique()
    n = len(df)

    for value in unique_values:
        subset = df[df[feature] == value]
        subset_var = subset[target].var()
        weight = len(subset) / n
        total_var += weight * subset_var

    return total_var


def questao2():
    target = 'price'
    features = [col for col in df_house.columns if col != target]

    s_values = {}
    for feature in features:
        s_values[feature] = calculate_sx(df_house, feature, target)

    s_df = pd.DataFrame.from_dict(s_values, orient='index', columns=['S(T,X)'])

    return s_df


def calculate_sdr(df, feature, target):
    total_var = df[target].var()
    s_value = calculate_sx(df, feature, target)
    sdr = total_var - s_value

    return sdr


def questao3():
    target = 'price'
    features = [col for col in df_house.columns if col != target]

    sdr_values = {}
    for feature in features:
        sdr_values[feature] = calculate_sdr(df_house, feature, target)

    sdr_df = pd.DataFrame.from_dict(sdr_values, orient='index', columns=['SDR(T,X)'])
    max_sdr_feature = sdr_df['SDR(T,X)'].idxmax()

    return max_sdr_feature


print("Questao1: ", questao1())
print("Questao2: ", questao2())
print("Questao3: ", questao3())
