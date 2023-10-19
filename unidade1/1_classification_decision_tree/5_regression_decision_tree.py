from src.utils import load_auto_dataset
from sklearn.model_selection import train_test_split

# Import DecisionTreeRegressor from sklearn.tree
from ___ import ___

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

df = load_auto_dataset()
X = df.drop(['mpg','origin'],axis=1)
y = df['mpg'].values


# Instantiate dt
dt = ___

X_train, X_test, y_train, y_test = ___

# Fit dt to the training set
____

# Compute y_pred
y_pred = ___

# Compute mse_dt
mse_dt =___

# Compute rmse_dt
rmse_dt = ___

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))