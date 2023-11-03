# PACKAGES -------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np


# FUNÇÕES --------------------------------------------------------------------------------------------------------------

# Função para calcular o índice de Gini
def calculate_gini(data, attribute, target):
    gini = 1
    attribute_values = data[attribute].unique()

    for value in attribute_values:
        # Calcula a probabilidade de um valor do atributo
        p_value = len(data[data[attribute] == value]) / len(data)
        # Calcula a probabilidade de 'Y' dado o valor do atributo
        p_target = len(data[(data[attribute] == value) & (data[target] == 'Y')]) / len(data[data[attribute] == value])
        # Atualiza o índice de Gini
        gini -= (p_target ** 2 + (1 - p_target) ** 2) * p_value

    return gini


# Função para calcular a entropia
def calculate_entropy(data, attribute, target):
    entropy = 0
    attribute_values = data[attribute].unique()

    for value in attribute_values:
        # Calcula a probabilidade de um valor do atributo
        p_value = len(data[data[attribute] == value]) / len(data)
        # Calcula a probabilidade de 'Y' dado o valor do atributo
        p_target = len(data[(data[attribute] == value) & (data[target] == 'Y')]) / len(data[data[attribute] == value])

        if p_target != 0:
            # Atualiza a entropia
            entropy -= p_value * p_target * np.log2(p_target)

    return entropy


# TESTE ----------------------------------------------------------------------------------------------------------------

# Dados
df = pd.DataFrame({
    'N1': ['Rainy', 'Cloudy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Rainy',
           'Sunny', 'Cloudy', 'Cloudy', 'Sunny', 'Rainy', 'Rainy', 'Sunny'],
    'N2': ['Medium', 'High', 'High', 'Medium', 'High', 'High', 'Low', 'Low',
           'Medium', 'Medium', 'High', 'High', 'Low', 'Medium'],
    'N3': ['East', 'West', 'East', 'East', 'West', 'East', 'East', 'West',
           'West', 'East', 'East', 'East', 'West', 'East'],
    'Class': ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N']
})

# Calcular o Gini e a entropia para cada atributo
target_attribute = 'Class'
attributes = df.drop(target_attribute, axis=1)

gini_scores = {}
entropy_scores = {}

for attribute in attributes:
    # Calcula o índice de Gini e a entropia para cada atributo
    gini_score = calculate_gini(df, attribute, target_attribute)
    entropy_score = calculate_entropy(df, attribute, target_attribute)

    # Armazena os resultados em dicionários
    gini_scores[attribute] = gini_score
    entropy_scores[attribute] = entropy_score

print("Índice de Gini:")
print(gini_scores)

print("\nEntropia:")
print(entropy_scores)

# Encontre os dois atributos com os menores valores de Gini e entropia
best_gini_attributes = sorted(gini_scores, key=gini_scores.get)[:2]
best_entropy_attributes = sorted(entropy_scores, key=entropy_scores.get)[:2]

print("\nMelhores atributos com base no índice de Gini:", best_gini_attributes)
print("Melhores atributos com base na entropia:", best_entropy_attributes)
