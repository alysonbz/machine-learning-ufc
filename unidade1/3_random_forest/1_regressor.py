import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from src.utils import bike_rental_dataset
from sklearn.model_selection import train_test_split

# Carregar o dataset
df = bike_rental_dataset()
X = df.drop(['count'], axis=1)
y = df['count'].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instanciar o RandomForestRegressor
rf = RandomForestRegressor(n_estimators=25, random_state=2)

# Ajustar o modelo ao conjunto de treino
rf.fit(X_train, y_train)

# Prever os rótulos do conjunto de teste
y_pred = rf.predict(X_test)

# Avaliar o RMSE do conjunto de teste
rmse_test = MSE(y_test, y_pred, squared=False)

# Imprimir o RMSE do conjunto de teste
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Criar uma série do pandas com as importâncias das features
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Ordenar as importâncias
importances_sorted = importances.sort_values()

# Desenhar um gráfico de barras horizontal com as importâncias ordenadas
importances_sorted.plot(kind='barh', color='skyblue')
plt.title('Features Importances')
plt.show()
