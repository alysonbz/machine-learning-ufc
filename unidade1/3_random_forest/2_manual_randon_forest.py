# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import clone
from sklearn.base import clone
from sklearn.utils import resample
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter
import random


class RandomForest:
    def __init__(self, base_estimator, n_estimators, max_features):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = []
        self.feature_subsets = []

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.models = []
        self.feature_subsets = []

        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample, sample_indices = resample(X_train, y_train, range(n_samples), n_samples=n_samples,
                                                          replace=True, random_state=SEED + i)

            # Randomly select max_features for this tree
            features = random.sample(range(n_features), self.max_features)
            self.feature_subsets.append(features)

            # Create a new decision tree and train it on the bootstrap sample
            model = clone(self.base_estimator)
            model.fit(X_sample[:, features], y_sample)
            self.models.append(model)

    def predict(self, X):
        # Collect predictions from each estimator
        predictions = []
        for model, features in zip(self.models, self.feature_subsets):
            pred = model.predict(X[:, features])
            predictions.append(pred)

        # Transpose the list of predictions to have each row corresponding to a sample
        predictions = list(zip(*predictions))

        # Use majority voting to determine the final prediction for each sample
        final_prediction = [Counter(row).most_common(1)[0][0] for row in predictions]

        return final_prediction


# Set seed for reproducibility
SEED = 1
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate rf
rf = RandomForest(base_estimator=dt, n_estimators=50, max_features=int(X_train.shape[1] ** 0.5))

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict test set labels
y_pred = rf.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_pred, y_test)

# Print acc_test
print('Test set accuracy: {:.3f}'.format(acc_test))
