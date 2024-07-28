# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier

from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN


class Voting_classifier:

    def __init__(self, base_estimators):
        self.base_estimators = base_estimators



    def fit(self, X_train, y_train):
        for name, estimator in self.base_estimators:
            estimator.fit(X_train,y_train)


    def predict(self, X):
        predictions = []
        for name, estimator in self.base_estimators:
            predictions.append(estimator.predict(X))

            final_predictions = []
            for i in range(len(X)):
                votes = [pred[i] for pred in predictions]
                final_predictions.append(max(set(votes), key=votes.count))

            return final_predictions


# Set seed for reproducibility
SEED = 1
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Instantiate bc
vc = Voting_classifier(base_estimators=classifiers)

# Fit bc to the training set
vc.fit(X_train, y_train)

# Predict test set labels
y_pred = vc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_pred, y_test)

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}'.format(acc_test))
