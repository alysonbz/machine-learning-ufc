import warnings
warnings.filterwarnings("ignore")
# Import functions to compute accuracy and split data
from __ import ___
from __ import __

# Import models, including VotingClassifier meta-model
from ___ import ____
from ____ import ___
from ____ import ____
from ____ import ___

from src.utils import indian_liver_dataset
from sklearn.preprocessing import StandardScaler


df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values

# Set seed for reproducibility
SEED=1

#spit the dataset
X_train, X_test, y_train, y_test = ____


# Instantiate lr - logisic regression
lr = ___

# Instantiate knn with 27 neighbors
knn = ---

# Instantiate dt - decision tree with min_sample_leaf 0.13 and random state SEED
dt = ___

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ___, ___]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in ___:
    # Fit clf to the training set
    ___

    # Predict y_pred
    y_pred = ___

    # Calculate accuracy
    accuracy =___

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))



# Instantiate a VotingClassifier vc
vc = ____

# Fit vc to the training set
____

# Evaluate the test set predictions
y_pred = ___

# Calculate accuracy score
accuracy = ____
print('Voting Classifier: {:.3f}'.format(accuracy))