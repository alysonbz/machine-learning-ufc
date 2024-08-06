from src.utils import indian_liver_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

SEED=1
df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=SEED)


# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=SEED)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt,n_estimators=180, random_state=1)

# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:,1]

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))