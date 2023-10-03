from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class Bagging:

    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.oob = None

    def fit(self, X_train, y_train):
        n_samples = X_train.shape[0]
        self.estimators = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]

            # Create a base estimator and fit it
            estimator = self.base_estimator
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)

        # Calculate OOB score
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for i in range(self.n_estimators):
            indices = np.setdiff1d(range(n_samples), indices)
            oob_predictions[indices] += self.estimators[i].predict(X_train[indices])
            oob_counts[indices] += 1

        self.oob = (oob_predictions / oob_counts >= 0.5).astype(int)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for i in range(self.n_estimators):
            predictions[:, i] = self.estimators[i].predict(X)

        return (predictions.mean(axis=1) >= 0.5).astype(int)

    def oob_score(self, y_train):
        return accuracy_score(y_train, self.oob)

# Set seed for reproducibility
SEED = 1

# Carregue os dados (você pode substituir esta parte pelo seu próprio carregamento de dados)
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values

# Divida os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = Bagging(base_estimator=dt, n_estimators=50)

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score(y_train)

# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
