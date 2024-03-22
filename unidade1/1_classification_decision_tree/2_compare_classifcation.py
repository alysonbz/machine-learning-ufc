#import LogisticRegression from sklearn.linear_model
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_classifier(clf,X_train, X_test, y_train, y_test):
    train_test_split(X,y, X_train,y_train,X_test,y_test)



df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]].values
y  = df_breast[['diagnosis']].values

# Instatiate logreg and decision tree
logreg = LogisticRegression()
dt = DecisionTreeClassifier()

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=3)

# Fit logreg to the training set
fit = dt.fit(X_train,y_train)

# call function to process dt

