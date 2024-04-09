# Import LogisticRegression from sklearn.linear_model
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_classifier(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc



df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]].values
y  = df_breast[['diagnosis']].values

# Instatiate logreg and decision tree
logreg = LogisticRegression()
dt = DecisionTreeClassifier()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Call function to process logistic regression classifier
logreg_acc = process_classifier(logreg, X_train, X_test, y_train, y_test)
print("Logistic Regression Accuracy:", logreg_acc)

# Call function to process decision tree classifier
dt_acc = process_classifier(dt, X_train, X_test, y_train, y_test)
print("Decision Tree Accuracy:", dt_acc)