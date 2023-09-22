from sklearn.linear_model import LogisticRegression
from src.utils import indian_liver_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt


random.seed(12)

df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)


lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

bias = np.mean(np.mean(y_test)-np.mean(y_pred))
print('Bias:', bias)

var_test = np.var(y_test)
var_pred = np.var(y_pred)
print('Variância Test:', var_test,
      '\nVariância Pred:',  var_pred)


# gerar nums aleatorios - 0.1 a 1 , 0.2 a 2
bias = np.array([random.uniform(0.1, 1.0) for _ in range(10)])
var = np.array([random.uniform(0.2, 2.0) for _ in range(10)])

erro = bias**2 + var

plt.scatter(erro, var)
plt.scatter(erro, bias)
plt.show()

