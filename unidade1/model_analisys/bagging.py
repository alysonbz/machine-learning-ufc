from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import DecisionTreeClassifier
____
# Import BaggingClassifier
___

# Set seed for reproducibility
SEED=1

df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)

# Instantiate dt
dt = ___

# Instantiate bc
bc = ___

# Fit bc to the training set
__

# Predict test set labels
y_pred = __

# Evaluate acc_test
acc_test =

print('Test set accuracy of bc: {:.2f}'.format(acc_test))