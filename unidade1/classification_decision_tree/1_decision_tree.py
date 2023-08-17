# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from src.utils import load_breast_cancer_dataset
from sklearn.model_selection import train_test_split

df_breast = load_breast_cancer_dataset()
X = df_breast[["diagnosis", "radius_mean","texture_mean","perimeter_mean","area_mean"]].values
y  = df_breast[['diagnosis']].values

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = ___


X_train, X_test, y_train, y_test =___

# Fit dt to the training set
___

# Predict test set labels
___

#print predicted labels
__