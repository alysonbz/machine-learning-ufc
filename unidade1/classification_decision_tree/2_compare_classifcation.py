# Import LogisticRegression from sklearn.linear_model
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_classifier(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    # Predict test set labels
    y_pred =  clf.predict(X_test)
    print(y_pred[0:10])
    # Compute test set accuracy
    acc = accuracy_score(y_pred, y_test)

    print("Test set accuracy: {:.2f}".format(acc))



df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]].values
y  = df_breast[['diagnosis']].values

# Instatiate logreg
logreg = LogisticRegression(random_state=1)
dt = DecisionTreeClassifier(max_depth=6, random_state=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
# Fit logreg to the training set

process_classifier(logreg,X_train, X_test, y_train, y_test)
process_classifier(dt,X_train, X_test, y_train, y_test)
