from src.utils import load_breast_cancer_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean"]].values
y  = df_breast[['diagnosis']].values

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth =6, random_state = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size =0.2,
                                                    stratify = y,
                                                    random_state =1)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)

#print predicted labels
print(f'predições: {y_pred}')

# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
#print the score
print("Test set accuracy: {:.2f}".format(acc))