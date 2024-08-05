import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/ufc/savim/machine-learning-ufc/AV1/plant_growth_data_pos.csv")

X = df.drop(['Growth_Milestone'],axis=1).values
y = df[['Growth_Milestone']].values

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X)

# Set seed for reproducibility
SEED=1

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y,random_state=1)

# Instantiate dt - decision tree with min_sample_leaf 0.13 and random state SEED
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Instantiate rf - Random Forest with max_depth=4 and random state SEED
rf = RandomForestClassifier(max_depth=4, random_state=SEED)

# Instantiate ada
ada = AdaBoostClassifier(estimator=dt,  n_estimators=180, random_state=SEED)

# Instantiate gb
gb = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_features=0.75, n_estimators=200,random_state=2)

# Instantiate sgbr
sgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)

# Define the list classifiers
classifiers = [('Decision Tree', dt), ('Random Forest',dt), ('Adaboost', ada), ('Gradient Boosting', gb), ('SGB', sgb)]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train,y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test,y_pred)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

