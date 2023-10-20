from sklearn.model_selection import train_test_split
# Import GradientBoostingRegressor
_____
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
from src.utils import bike_rental_dataset

df = bike_rental_dataset()
X = df.drop(['count'],axis = 1)
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# Instantiate gb
gb = ___

# Fit gb to the training set
___

# Predict test set labels
y_pred = __

# Compute MSE
mse_test = __

# Compute RMSE
rmse_test = ___

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(___))