import pandas as pd
import numpy as np

df = pd.read_excel("img2.xlsx")

# Funções de entropia e índice de Gini para calcular a impureza
def entropy(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def gini(y):
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    return 1 - np.sum([p**2 for p in probabilities])

# Conversão
y_numeric = df['play'].replace({'yes': 1, 'no': 0}).values

# Calcular a entropia e o índice de Gini inicial para y
initial_entropy = entropy(y_numeric)
initial_gini = gini(y_numeric)

# Função para calcular o ganho de pureza
def calculate_gain(df, attribute, initial_impurity, impurity_function):
    values = df[attribute].unique()
    weighted_impurity = 0
    for value in values:
        subset = df[df[attribute] == value]['play'].replace({'yes': 1, 'no': 0}).values
        weighted_impurity += (len(subset) / len(df)) * impurity_function(subset)
    return initial_impurity - weighted_impurity

# Calcular os ganhos de pureza para cada atributo
gains_entropy = {}
gains_gini = {}
columns_to_transform = df.columns.drop(['play', 'day'])

for col in columns_to_transform:
    gains_entropy[col] = calculate_gain(df, col, initial_entropy, entropy)
    gains_gini[col] = calculate_gain(df, col, initial_gini, gini)

# Ordenar os atributos pelos ganhos de pureza
sorted_gains_entropy = sorted(gains_entropy.items(), key=lambda x: x[1], reverse=True)
sorted_gains_gini = sorted(gains_gini.items(), key=lambda x: x[1], reverse=True)

# Imprimir resultados
print("Entropia inicial:", initial_entropy)
print("Índice de Gini inicial:", initial_gini)
print("\nGanho de pureza usando Entropia:")
for attr, gain in sorted_gains_entropy:
    print(f"Atributo: {attr}, Ganho: {gain}")
print("\nGanho de pureza usando Índice de Gini:")
for attr, gain in sorted_gains_gini:
    print(f"Atributo: {attr}, Ganho: {gain}")

# Imprimir o atributo com maior ganho de pureza
best_attribute_entropy = sorted_gains_entropy[0]
best_attribute_gini = sorted_gains_gini[0]
print(f"\nAtributo com maior ganho usando Entropia: {best_attribute_entropy[0]}, Ganho: {best_attribute_entropy[1]}")
print(f"Atributo com maior ganho usando Índice de Gini: {best_attribute_gini[0]}, Ganho: {best_attribute_gini[1]}")