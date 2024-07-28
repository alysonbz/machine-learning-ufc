# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier
from sklearn.utils import resample
import numpy as np

from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone


class Bagging:

    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.oob_samples = []
        self.oob_predictions = []

    def fit(self, X_train, y_train):
        n_samples = X_train.shape[0]
        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = resample(X_train, y_train, n_samples=n_samples, replace=True, random_state=SEED)
            estimator = clone(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)
            # Calculate OOB samples
            mask = ~np.isin(range(n_samples), np.unique(resample(range(n_samples), replace=True, random_state=SEED)))
            oob_sample = X_train[mask]
            oob_prediction = estimator.predict(oob_sample)
            self.oob_samples.append(mask)
            self.oob_predictions.append(oob_prediction)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(X)
        return np.round(predictions.mean(axis=1)).astype(int)

    def oob_score(self, X_train, y_train):
        oob_predictions = np.zeros(X_train.shape[0])
        for i, mask in enumerate(self.oob_samples):
            oob_predictions[mask] += self.oob_predictions[i]
        oob_predictions /= self.n_estimators
        oob_predictions = np.round(oob_predictions).astype(int)
        return accuracy_score(y_train, oob_predictions)

SEED = 1
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

dt = DecisionTreeClassifier(random_state=1)
bc = Bagging(base_estimator=dt, n_estimators=50)

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_oob = bc.oob_score(X_train, y_train)
acc_test = accuracy_score(y_pred, y_test)

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
