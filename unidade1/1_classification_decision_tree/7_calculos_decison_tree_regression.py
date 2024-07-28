import pandas as pd
import numpy as np
from src.utils import load_house_price_dataset

df_house = load_house_price_dataset()


def questao1():
    price = df_house['price']
    count = price.count()
    average = price.mean()
    std_dev = price.std()
    coeff_var = std_dev / average
    return count, average, std_dev, coeff_var


def questao2():
    target = df_house['price']
    features = df_house.drop(columns=['price'])

    # Convert categorical variables to numerical
    features = pd.get_dummies(features, drop_first=True)

    S_T_X = {}

    for column in features.columns:
        values = features[column]
        mean_target = target.mean()
        S_T_X[column] = np.sum((target - mean_target) ** 2)

    df_S_T_X = pd.DataFrame(S_T_X.items(), columns=['Attribute', 'S(T,X)'])
    return df_S_T_X


def questao3():
    target = df_house['price']
    features = df_house.drop(columns=['price'])

    # Convert categorical variables to numerical
    features = pd.get_dummies(features, drop_first=True)

    SDR_T_X = {}

    for column in features.columns:
        values = features[column]
        mean_target = target.mean()
        overall_std = target.std()
        weighted_std = np.sum((values - values.mean()) ** 2)
        SDR_T_X[column] = overall_std - (weighted_std / len(values))

    df_SDR_T_X = pd.DataFrame(SDR_T_X.items(), columns=['Attribute', 'SDR(T,X)'])
    best_attribute = df_SDR_T_X.loc[df_SDR_T_X['SDR(T,X)'].idxmax()]

    return best_attribute['Attribute']


print("Questao1: ", questao1())
print("Questao2: ")
print(questao2())
print("Questao3: ",questao3())