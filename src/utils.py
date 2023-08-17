import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_breast_cancer_dataset():
    df = pd.read_csv('../dataset/data.csv')
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])
    return df