from src.utils import load_auto_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE

def compute_regressor_rmse(reg, X_train, X_test, y_train, y_test):
    # Fit reg to the training set
    reg.fit(X_train, y_train)
    # Compute y_pred
    y_pred = reg.predict(X_test)
    # Compute mse
    mse = MSE(y_test, y_pred)
    # Compute rmse
    rmse = mse**(1/2)
    return rmse

# Load dataset
df = load_auto_dataset()
X = df.drop(['mpg','origin'], axis=1)
y = df['mpg'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Instantiate decision tree regressor with max_depth=5
dt = DecisionTreeRegressor(max_depth=5)
lr = LinearRegression()

# Compute RMSE for both models
rmse_dt = compute_regressor_rmse(dt, X_train, X_test, y_train, y_test)
rmse_lr = compute_regressor_rmse(lr, X_train, X_test, y_train, y_test)

# Print RMSE for linear regression
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print RMSE for decision tree regression
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))


## resultados 5.12 e 4.37