import pandas as pd
from model_selection import Best_Model, get_classification_report
from sklearn.model_selection import train_test_split

# Leitura dos dados
df = pd.read_csv('df_final.csv')
columns_drop = ['True','Violence ']
X = df.drop(columns_drop, axis=1)
y = df['Violence ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_model, best_model_name = Best_Model(X_train, X_test, y_train, y_test)

# Obter o classification report e matriz de confusão do melhor modelo
report, matrix = get_classification_report(best_model, X_test, y_test)

# Exibir resultados
print(f"\nMelhor modelo: {best_model_name}")
print("\nClassification Report:")
print(report)
print("\nMatriz de Confusão:")
print(matrix)
