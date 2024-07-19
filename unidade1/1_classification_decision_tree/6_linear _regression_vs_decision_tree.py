from src.utils import load_auto_dataset
from sklearn.model_selection import train_test_split
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
# Import LinearRegression
from sklearn.linear_model import LinearRegression
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

def compute_regressor_rmse(reg, X_train, X_test, y_train, y_test):

    # Fit reg to the training set
    reg.fit(X_train, y_train)

    # Compute y_pred
    y_pred = reg.predict(X_test)

    # Compute mse
    mse = MSE(y_test, y_pred)

    # Compute rmse
    rmse = mse ** 0.5

    return rmse

df = load_auto_dataset()
X = df.drop(['mpg', 'origin'], axis=1)
y = df['mpg'].values

# Instantiate dt and lr
dt = DecisionTreeRegressor(max_depth=8, random_state=42)
lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Call compute_regressor_rmse function for the two regressors
rmse_dt = compute_regressor_rmse(dt, X_train, X_test, y_train, y_test)
rmse_lr = compute_regressor_rmse(lr, X_train, X_test, y_train, y_test)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
