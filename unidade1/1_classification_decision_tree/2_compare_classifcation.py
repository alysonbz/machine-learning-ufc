# Import LogisticRegression from sklearn.linear_model
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_classifier(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


df_breast = load_breast_cancer_dataset()

X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]].values
y  = df_breast[['diagnosis']].values

# Instatiate logreg and decision tree
logreg = LogisticRegression()
dt = DecisionTreeClassifier()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Fit logreg to the training set

# call function to process log_reg
log_reg = process_classifier(logreg, X_train,X_test, y_train, y_test)
print(f"Accuracy with log reg, {log_reg}")
# call function to process dt
dt_class = process_classifier(dt, X_train,X_test, y_train, y_test)
print(f"Accuracy with log reg, {dt_class}")