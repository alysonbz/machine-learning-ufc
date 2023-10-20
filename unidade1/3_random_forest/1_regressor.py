import pandas as pd
import matplotlib.pyplot as plt
# Import RandomForestRegressor
____

# Import mean_squared_error as MSE
from .____ import ____ as ____

from src.utils import bike_rental_dataset
from sklearn.model_selection import train_test_split

df = bike_rental_dataset()
X = df.drop(['count'],axis = 1)
y = df['count'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# Instantiate rf
rf = ____(n_estimators=____,
          random_state=2)

# Fit rf to the training set
____.____(____, ____)


# Predict the test set labels
y_pred = ____.____(____)

# Evaluate the test set RMSE
rmse_test = ____

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))


# Create a pd.Series of features importances
importances = ___

# Sort importances
importances_sorted =___

# Draw a horizontal barplot of importances_sorted
____
plt.title('Features Importances')
plt.show()