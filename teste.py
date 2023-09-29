from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    stractfy = y)

df.fit(x_train, y_train)

y_pred = df.predict(x_test)

accuracy_score(y_test, y_pred)