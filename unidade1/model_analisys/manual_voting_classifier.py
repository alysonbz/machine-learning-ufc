from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

class Voting_classifier:

    def __init__(self, base_estimators):
        self.base_estimators = base_estimators

    def fit(self, X_train, y_train):
        for _, estimator in self.base_estimators:
            estimator.fit(X_train, y_train)

    def predict(self, X):
        # Realize previsões com todos os estimadores base
        predictions = [estimator.predict(X) for _, estimator in self.base_estimators]
        # Calcule a previsão final por voto majoritário (hard voting)
        final_predictions = np.array([np.bincount(prediction).argmax() for prediction in np.array(predictions).T])
        return final_predictions

# Set seed for reproducibility
SEED = 1
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values

# Divida o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate lr (Logistic Regression)
lr = LogisticRegression(random_state=SEED)

# Instantiate knn (K Nearest Neighbors)
knn = KNN(n_neighbors=3)

# Instantiate dt (Decision Tree)
dt = DecisionTreeClassifier(random_state=SEED)

# Instantiate bagging_classifier with Decision Tree
bagging_classifier = BaggingClassifier(base_estimator=dt, n_estimators=10, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt), ('Bagging Classifier', bagging_classifier)]

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
