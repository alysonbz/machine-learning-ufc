from src.utils import load_auto_dataset
from sklearn.model_selection import train_test_split

# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

df = load_auto_dataset()
X = df.drop(['mpg','origin'],axis=1)
y = df['mpg'].values


# Instantiate dt
dt = DecisionTreeRegressor(random_state=3,
                           max_depth=6,
                           min_samples_leaf=0.13)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Fit dt to the training set
dt.fit(X_train, y_train)

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_test, y_pred)

# Compute rmse_dt
rmse_dt = mse_dt*(1/2)

# print rmse_dt
print("RMSE: {:.2f}".format(rmse_dt))
