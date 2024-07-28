from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.utils import indian_liver_dataset
import numpy as np

class VotingClassifier:

    def __init__(self, base_estimators):
        self.base_estimators = base_estimators
        self.classifiers = []

    def fit(self, X_train, y_train):
        for _, clf in self.base_estimators:
            clf.fit(X_train, y_train)
            self.classifiers.append(clf)

    def predict(self, X):
        predictions = np.asarray([clf.predict(X) for clf in self.classifiers])
        predictions = predictions.T
        majority_votes = [np.bincount(pred).argmax() for pred in predictions]
        return np.array(majority_votes)

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

# Instantiate the VotingClassifier
vc = VotingClassifier(base_estimators=classifiers)

# Fit the VotingClassifier to the training set
vc.fit(X_train, y_train)

# Predict test set labels
y_pred = vc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_pred, y_test)

# Print acc_test
print('Test set accuracy: {:.3f}'.format(acc_test))
