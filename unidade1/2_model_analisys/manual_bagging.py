# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier

from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class Bagging:

    def __init__(self,base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.oob = None


    def fit(self,X_train,y_train):
        pass

    def predict(self,X):
        pass

    def oob_score(self):
        pass

# Set seed for reproducibility
SEED=1
df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)
# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = Bagging(base_estimator=dt, n_estimators=50)

# Fit bc to the training set
bc.fit(X_train ,y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score

# Evaluate acc_test
acc_test = accuracy_score(y_pred,y_test)

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
