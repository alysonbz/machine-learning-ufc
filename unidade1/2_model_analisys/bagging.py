from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

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

# Instantiate the BaggingClassifier
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=SEED)

# Fit the BaggingClassifier to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate the accuracy on the test set
acc_test = accuracy_score(y_test, y_pred)

print('Test set accuracy of bc: {:.2f}'.format(acc_test))
