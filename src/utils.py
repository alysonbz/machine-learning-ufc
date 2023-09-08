import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import mlxtend.plotting
import numpy as np

def load_breast_cancer_dataset():
    df = pd.read_csv('../dataset/data.csv')
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])
    return df

def plot_labeled_decision_regions(X_test, y_test, clfs):
    y_test = y_test[:,0]
    for clf in clfs:
        mlxtend.plotting.plot_decision_regions(np.array(X_test), np.array(y_test), clf=clf, legend=2,filler_feature_values=4)

        plt.ylim((0, 0.2))

        # Adding axes annotations
        plt.xlabel(X_test.columns[0])
        plt.ylabel(X_test.columns[1])
        plt.title(str(clf).split('(')[0])
        plt.show()


def load_auto_dataset():
    df = pd.read_csv('../dataset/auto.csv')
    return df