from sklearn.model_selection import train_test_split
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
from src.utils import bike_rental_dataset

df = bike_rental_dataset()
X = df.drop(['count'],axis = 1)
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# Instantiate gb
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, subsample=0.9,max_features=0.75, random_state=2)

# Fit gb to the training set
gb.fit(X_train, y_train)

# Predict test set labels
y_pred = gb.predict(X_test)

# Compute MSE
mse_test = MSE(y_test, y_pred)**(1/2)

# Compute RMSE
rmse_test = mse_test ** 0.5

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))