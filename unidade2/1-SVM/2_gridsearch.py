from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC


diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instantiate an RBF SVM
svm = SVC(kernel="rbf")

# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(param_grid=parameters,estimator=svm)

#fit the searcher
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)


