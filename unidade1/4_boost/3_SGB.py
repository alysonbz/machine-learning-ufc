from sklearn.model_selection import train_test_split
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from src.utils import bike_rental_dataset

df = bike_rental_dataset()
X = df.drop(['count'],axis = 1)
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth= 1,
                                 subsample=0.8,
                                 max_features=0.2,
                                 n_estimators=300,
                                 random_state=SEED)

# Fit sgbr to the training set
sgbr.fit(X_train,y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

# Compute MSE
mse_test = MSE(y_test,y_pred)

# Compute RMSE
rmse_test = MSE(y_test,y_pred)**(1/2)

# Print RMSE
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))