from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


columns = ["No Taste or Smell", "Fever", "Cough", "Coronavirus"]

data = [
    ["Yes", "Yes", "No", "Yes"],
    ["No", "No", "Yes", "No"],
    ["No", "Yes", "Yes", "Yes"],
    ["Yes", "No", "No", "Yes"],
    ["Yes", "No", "Yes", "Yes"],
    ["No", "No", "No", "No"],
    ["Yes", "Yes", "Yes", "Yes"],
    ["Yes", "No", "No", "No"],
    ["No", "No", "Yes", "No"],
    ["Yes","Yes","No","Yes"]
]

data_dict = {columns[i]: [row[i] for row in data] for i in range(len(columns))}
df = pd.DataFrame(data_dict)

le = LabelEncoder()
for column in columns:
    df[column] = le.fit_transform(df[column])

X = df.drop("Coronavirus", axis=1)
y = df["Coronavirus"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=1)
clf_gini.fit(X,y)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1)
clf_entropy.fit(X,y)

def mostrar_pontos_divisao(clf, criterion_name):
    print(f"Pontos de divisão para {criterion_name}:")
    for feature, threshold in zip(clf.tree_.feature, clf.tree_.threshold):
        if feature != -2:
            print(f"Feature: {columns[feature]}, Threshold: {threshold}")

mostrar_pontos_divisao(clf_gini, "Gini")
mostrar_pontos_divisao(clf_entropy, "Entropia")