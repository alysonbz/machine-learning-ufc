import pandas as pd
import matplotlib.pyplot as plt
# Import RandomForestRegressor
from sklearn.ensemble import  RandomForestRegressor

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

from src.utils import bike_rental_dataset
from sklearn.model_selection import train_test_split

df = bike_rental_dataset()
X = df.drop(['count'],axis = 1)
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# Instantiate rf
rf = RandomForestRegressor(n_estimators=400,
          random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)

# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test,y_pred)**(1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# Create a pd.Series of features importances
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen');
plt.title('Features Importances')
plt.show()