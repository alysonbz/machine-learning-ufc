from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.utils import resample
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter

class Bagging:
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        self.oob_scores = []

    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        self.models = []
        self.oob_scores = []

        for i in range(self.n_estimators):
            X_sample, y_sample, sample_indices = resample(X_train, y_train, range(n_samples), n_samples=n_samples, replace=True, random_state=SEED + i)
            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)

            oob_indices = [idx for idx in range(n_samples) if idx not in sample_indices]
            if len(oob_indices) > 0:
                X_oob = [X_train[idx] for idx in oob_indices]
                y_oob = [y_train[idx] for idx in oob_indices]
                oob_pred = model.predict(X_oob)
                self.oob_scores.append(accuracy_score(y_oob, oob_pred))

        self.oob = sum(self.oob_scores) / len(self.oob_scores) if self.oob_scores else None

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]

        predictions = list(zip(*predictions))

        final_prediction = [Counter(row).most_common(1)[0][0] for row in predictions]

        return final_prediction

    def oob_score(self):
        return self.oob


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

# Instantiate bc
bc = Bagging(base_estimator=dt, n_estimators=50)

# Fit bc to the training set
bc.fit(X_train,y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score()

# Evaluate acc_test
acc_test = accuracy_score(y_pred, y_test)

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
