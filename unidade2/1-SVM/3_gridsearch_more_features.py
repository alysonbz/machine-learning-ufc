from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC


diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instantiate an RBF SVM
svm = ___

# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = ___

#fit the searcher
___

# Report the best parameters and the corresponding score
print("Best CV params", ___)
print("Best CV accuracy", ___)

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test,y_test))