# Import LogisticRegression from sklearn.linear_model
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_classifier(clf, X_train, X_test, y_train, y_test):
    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# Load the breast cancer dataset
df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]].values
y = df_breast[['diagnosis']].values

# Instantiate LogisticRegression and DecisionTreeClassifier
logreg = LogisticRegression(random_state=1)
dt = DecisionTreeClassifier(random_state=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit logreg to the training set and evaluate it
logreg_accuracy = process_classifier(logreg, X_train, X_test, y_train, y_test)
print(f'Logistic Regression accuracy: {logreg_accuracy:.3f}')

# Fit dt to the training set and evaluate it
dt_accuracy = process_classifier(dt, X_train, X_test, y_train, y_test)
print(f'Decision Tree accuracy: {dt_accuracy:.3f}')
