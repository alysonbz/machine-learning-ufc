import pandas as pd
from sklearn.model_selection import train_test_split
from RandomClassifier import ManualRandomClassifier

rc = ManualRandomClassifier()
df = pd.read_csv("adult_1.1.csv")

# variáveis
X = df.drop('income', axis=1)
y = df['income']

# treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# DecisionTreeClassifier
clf_tree = ManualRandomClassifier(base_classifier='decision_tree', n_estimators=100, max_features='sqrt', max_depth=6)
acc_tree = clf_tree.train_evaluate(X_train, y_train, X_test, y_test)
print(f"Accuracy with Decision Tree: {acc_tree}")

# KNeighborsClassifier
clf_knn = ManualRandomClassifier(base_classifier='knn', n_estimators=100, max_features='sqrt', n_neighbors=5)
acc_knn = clf_knn.train_evaluate(X_train, y_train, X_test, y_test)
print(f"Accuracy with KNN: {acc_knn}")