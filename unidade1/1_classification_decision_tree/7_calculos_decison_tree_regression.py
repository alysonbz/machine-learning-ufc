from src.utils import load_house_price_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load house price dataset
df_house = load_house_price_dataset()

# Separate features and target variable
X = df_house.drop(columns=['SalePrice'])
y = df_house['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define and train the decision tree regressor
def train_decision_tree_regression(X_train, y_train):
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)
    return dt_regressor


# Function to answer question 1
def questao1():
    # Train the decision tree regressor
    dt_regressor = train_decision_tree_regression(X_train, y_train)
    # Evaluate the model on the training set
    train_mse = mean_squared_error(y_train, dt_regressor.predict(X_train))
    return train_mse


# Function to answer question 2
def questao2():
    # Train the decision tree regressor
    dt_regressor = train_decision_tree_regression(X_train, y_train)
    # Evaluate the model on the testing set
    test_mse = mean_squared_error(y_test, dt_regressor.predict(X_test))
    return test_mse


# Function to answer question 3
def questao3():
    # Train the decision tree regressor using max_depth=5
    dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_regressor.fit(X_train, y_train)
    # Evaluate the model on the testing set
    test_mse_depth5 = mean_squared_error(y_test, dt_regressor.predict(X_test))

    # Train the decision tree regressor using max_depth=None (fully grown tree)
    dt_regressor_full = DecisionTreeRegressor(random_state=42)
    dt_regressor_full.fit(X_train, y_train)
    # Evaluate the fully grown tree on the testing set
    test_mse_full = mean_squared_error(y_test, dt_regressor_full.predict(X_test))

    # Calculate the ratio of MSE of depth=5 tree to fully grown tree
    ratio_mse = test_mse_depth5 / test_mse_full
    return ratio_mse


# Call the functions and print the results
print("Questao1 (Training MSE): ", questao1())
print("Questao2 (Testing MSE): ", questao2())
print("Questao3 (Ratio of Testing MSE between depth=5 and fully grown tree): ", questao3())
