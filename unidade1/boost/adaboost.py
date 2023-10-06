from src.utils import indian_liver_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Import DecisionTreeClassifier
__

# Import AdaBoostClassifier
__

# Import roc_auc_score
__

SEED=1
df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=SEED)


# Instantiate dt
dt = __

# Instantiate ada
ada = __

# Fit ada to the training set
__

# Compute the probabilities of obtaining the positive class
y_pred_proba = ___

# Evaluate test-set roc_auc_score
ada_roc_auc = __

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))