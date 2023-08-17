from src.utils import load_breast_cancer_dataset
from sklearn.model_selection import train_test_split

# Import DecisionTreeClassifier from sklearn.tree
_____

df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean"]].values
y  = df_breast[['diagnosis']].values

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = ___


X_train, X_test, y_train, y_test =___

# Fit dt to the training set
___

# Predict test set labels
y_pred =

#print predicted labels
__

# Compute test set accuracy
acc = ___
#print the score
print("Test set accuracy: {:.2f}".format(acc))