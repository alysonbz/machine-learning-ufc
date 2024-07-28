from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from src.utils import bike_rental_dataset

# Carregar o dataset
df = bike_rental_dataset()
X = df.drop(['count'], axis=1)
y = df['count'].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instanciar o GradientBoostingRegressor
sgbr = GradientBoostingRegressor(n_estimators=100, random_state=2)

# Ajustar o modelo ao conjunto de treino
sgbr.fit(X_train, y_train)

# Prever os rótulos do conjunto de teste
y_pred = sgbr.predict(X_test)

# Calcular o MSE
mse_test = MSE(y_test, y_pred)

# Calcular o RMSE
rmse_test = mse_test ** 0.5

# Imprimir o RMSE
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
