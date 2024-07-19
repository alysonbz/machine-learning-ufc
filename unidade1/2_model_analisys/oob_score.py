# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
SEED = 1

# Load the dataset
df = indian_liver_dataset()

# Prepare the feature and target variables
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate the DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=SEED)

# Instantiate the BaggingClassifier with OOB score enabled
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=SEED, oob_score=True)

# Fit the BaggingClassifier to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate the test set accuracy
acc_test = accuracy_score(y_test, y_pred)

# Evaluate the OOB accuracy
acc_oob = bc.oob_score_

# Print test set accuracy and OOB accuracy
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
