
from src.utils import load_auto_dataset
from sklearn.model_selection import train_test_split
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

def compute_regressor_rmse(reg,X_train, X_test,y_train,y_test):

    # Fit dt to the training set
    ____

    # Compute y_pred
    y_pred = ___

    # Compute mse_dt
    mse  = ___

    # Compute rmse_dt
    rmse = ___

    return rmse

df = load_auto_dataset()
X = df.drop(['mpg','origin'],axis=1)
y = df['mpg'].values

# Instantiate dt and lr
dt = ___
lr = ___

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# call compute_regressor_rmse function for the two regressors
rmse_dt = ___
rmse_lr = ___

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))

