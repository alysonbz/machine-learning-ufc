from sklearn.model_selection import train_test_split
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
from src.utils import bike_rental_dataset

SEED = 2
df = bike_rental_dataset()
X = df.drop(['count'],axis = 1)
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)

# Instantiate gb
gb = GradientBoostingClassifier(n_estimators=20,max_depth=1,random_state=2)

# Fit gb to the training set00,
gb.fit(X_train,y_train)

# Predict test set labels
y_pred = gb.predict_proba(X_test)

# Compute MSE
mse_test = MSE(y_test,y_pred)

# Compute RMSE
rmse_test = mse_test**(1/2)

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))