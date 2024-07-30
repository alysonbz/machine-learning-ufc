from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier
from  sklearn.ensemble import BaggingClassifier

# Set seed for reproducibility
SEED=1

df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)

# Instantiate dt
dt = DecisionTreeClassifier()

# Instantiate bc
bc = BaggingClassifier()

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_test,y_pred)

print('Test set accuracy of bc: {:.2f}'.format(acc_test))