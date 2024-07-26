from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Voting_classifier:

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator


    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        pass

  def __init__(self, classifiers):







    self.classifiers = classifiers

  def fit(self, X_train, y_train):





    for name, clf in self.classifiers:
      clf.fit(X_train, y_train)

  def predict(self, X):







    predictions = [clf.predict(X) for _, clf in self.classifiers]
    # Majority vote based on class occurrences
    from collections import Counter
    class_counts = Counter([pred for pred_list in predictions for pred in pred_list])
    return class_counts.most_common(1)[0][0]  # Return most frequent class

# Set seed for reproducibility
SEED = 1
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Instantiate vc
vc = VotingClassifier(classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Predict test set labels
y_pred = vc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_pred, y_test)

# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(acc_test))
