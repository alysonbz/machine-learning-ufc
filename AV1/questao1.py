import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Carregar o DataFrame a partir do arquivo Excel
df = pd.read_excel("img1.xlsx")
print(df.head())

# Inicializar o LabelEncoder
le = LabelEncoder()

# Aplicar o LabelEncoder a cada coluna categórica
df['outlook'] = le.fit_transform(df['outlook'])
df['temperature'] = le.fit_transform(df['temperature'])
df['humidity'] = le.fit_transform(df['humidity'])
df['windy'] = le.fit_transform(df['windy'])
df['play_golf'] = le.fit_transform(df['play_golf'])

# Exibir o DataFrame mapeado
print(df.head())

# Calcular a entropia
def entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Calcular o índice de Gini
def gini(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return 1 - np.sum([p**2 for p in probabilities])

# Calcular a entropia e o índice de Gini inicial
attributes = ['outlook', 'temperature', 'humidity', 'windy']
y = df['play_golf'].values
initial_entropy = entropy(y)
initial_gini = gini(y)

print(f"Entropia inicial: {initial_entropy}")
print(f"Índice de Gini inicial: {initial_gini}")

# Calcular o ganho de pureza para cada atributo
def calculate_gain(df, attribute, initial_impurity, impurity_function):
    values = df[attribute].unique()
    weighted_impurity = 0
    for value in values:
        subset = df[df[attribute] == value]['play_golf'].values
        weighted_impurity += (len(subset) / len(df)) * impurity_function(subset)
    return initial_impurity - weighted_impurity

# Calcular os ganhos de pureza para cada atributo usando entropia e índice de Gini
gains_entropy = {}
gains_gini = {}

for col in attributes:
    gains_entropy[col] = calculate_gain(df, col, initial_entropy, entropy)
    gains_gini[col] = calculate_gain(df, col, initial_gini, gini)

# Ordenar os atributos pelos ganhos de pureza
sorted_gains_entropy = sorted(gains_entropy.items(), key=lambda x: x[1], reverse=True)
sorted_gains_gini = sorted(gains_gini.items(), key=lambda x: x[1], reverse=True)

print("Ganho de pureza usando Entropia:")
for attr, gain in sorted_gains_entropy:
    print(f"Atributo: {attr}, Ganho: {gain}")

print("\nGanho de pureza usando Índice de Gini:")
for attr, gain in sorted_gains_gini:
    print(f"Atributo: {attr}, Ganho: {gain}")

# Determinar as duas melhores possibilidades de nó raiz
best_entropy_attributes = sorted_gains_entropy[:2]
best_gini_attributes = sorted_gains_gini[:2]

print("\nMelhores atributos com Entropia:")
for attr, gain in best_entropy_attributes:
    print(f"Atributo: {attr}, Ganho: {gain}")

print("\nMelhores atributos com Índice de Gini:")
for attr, gain in best_gini_attributes:
    print(f"Atributo: {attr}, Ganho:{gain}")
