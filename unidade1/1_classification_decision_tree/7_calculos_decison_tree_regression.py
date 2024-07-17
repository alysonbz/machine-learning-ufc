import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from src.utils import load_house_price_dataset

df_house = load_house_price_dataset()


def questao1():

    count = df_house['price'].count()
    average = df_house['price'].mean()
    std_dev = df_house['price'].std()
    coef_var = std_dev / average

    return count, average, std_dev, coef_var





def questao2():
    for col in df_house.columns:
        if df_house[col].dtype == 'object':
            df_house[col] = pd.Categorical(df_house[col]).codes

    regressor = DecisionTreeRegressor()

    results = pd.DataFrame(columns=['Feature', 'S(T,X)'])

    for col in df_house.columns:
        if col != 'price':
            X = df_house[[col]]
            y = df_house['price']

            regressor.fit(X, y)

            importance = regressor.feature_importances_[0]

            results = results.append({'Feature': col, 'S(T,X)': importance}, ignore_index=True)

    return results







def questao3():
    # Calculate initial standard deviation of 'price'
    sigma_T = np.std(df_house['price'])

    # Initialize a DataFrame to store results
    results_sdr = pd.DataFrame(columns=['Feature', 'SDR(T,X)'])

    # Calculate SDR for each feature (column) except 'price'
    for col in df_house.columns:
        if col == 'price':
            continue

        # Calculate weighted standard deviation after partitioning by each feature
        weighted_sigma_T_X = 0.0
        for value in df_house[col].unique():
            subset_prices = df_house.loc[df_house[col] == value, 'price']
            weight = len(subset_prices) / len(df_house)
            weighted_sigma_T_X += weight * np.std(subset_prices)

        # Calculate SDR(T,X) = sigma_T - weighted_sigma_T_X
        sdr = sigma_T - weighted_sigma_T_X

        # Append results to DataFrame
        results_sdr = results_sdr.append({'Feature': col, 'SDR(T,X)': sdr}, ignore_index=True)

    # Find the feature with the highest SDR(T,X)
    max_sdr_feature = results_sdr.loc[results_sdr['SDR(T,X)'].idxmax(), 'Feature']

    return max_sdr_feature




print("Questao1: ",questao1())
print("Questao2: ",questao2())
print("Questao3: ",questao3())