
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.utils import load_breast_cancer_dataset

df = load_breast_cancer_dataset()

X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bias = []
variance = []

# Tamanhos dos subconjuntos de treino
train_sizes = np.linspace(0.2, 0.8, 10)

for train_size in train_sizes:
    mse_train = []
    mse_test = []
    for _ in range(10):
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_subset, y_train_subset)

        y_train_pred = model.predict(X_train_subset)

        y_test_pred = model.predict(X_test)

        mse_train.append(mean_squared_error(y_train_subset, y_train_pred))

        mse_test.append(mean_squared_error(y_test, y_test_pred))

    bias.append(np.mean(mse_test))
    variance.append(np.mean(mse_train))


plt.figure(figsize=(10, 6))
plt.plot(train_sizes, bias, label='Bias', marker='o')
plt.plot(train_sizes, variance, label='Variance', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.title('Bias-Variance Trade-off for Decision Tree Classifier')
plt.legend()
plt.grid(True)
plt.show()
