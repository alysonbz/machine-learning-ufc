# Import LogisticRegression from sklearn.linear_model
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_classifier(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'A acuracia é: {acc}')


df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]].values
y  = df_breast[['diagnosis']].values

# Instatiate logreg and decision tree
logreg = LogisticRegression(random_state=42)
dt = DecisionTreeClassifier(max_depth=4, min_samples_split=0.1, random_state=3)


# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# call function to process log_reg
process_classifier(logreg, X_train, X_test, y_train.ravel(), y_test.ravel())
# call function to process dt
process_classifier(dt, X_train, X_test, y_train.ravel(), y_test.ravel())

"""y_train.ravel() é usado para garantir que os rótulos de treinamento estejam no formato correto (1D array) 
antes de passá-los para os classificadores."""