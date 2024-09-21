import pandas as pd
from math import log2

df = pd.read_csv('bv')

# Funções para cálculo da Entropia e Ganho de Informação
def calculate_entropy(data, target_column):
    target_values = data[target_column].value_counts(normalize=True)
    entropy = -sum(p * log2(p) for p in target_values if p > 0)
    return entropy

def calculate_entropy_attribute(data, attribute, target_column):
    attribute_values = data[attribute].value_counts(normalize=True)
    conditional_entropy = 0
    for value in attribute_values.index:
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_column)
        conditional_entropy += attribute_values[value] * subset_entropy
    return conditional_entropy

def calculate_information_gain(data, attribute, target_column):
    total_entropy = calculate_entropy(data, target_column)
    attribute_entropy = calculate_entropy_attribute(data, attribute, target_column)
    information_gain = total_entropy - attribute_entropy
    return information_gain

# Funções para cálculo do Índice de Gini e Ganho de Gini
def calculate_gini_index(data, target_column):
    target_values = data[target_column].value_counts(normalize=True)
    gini = 1 - sum(target_values ** 2)
    return gini

def calculate_gini_attribute(data, attribute, target_column):
    attribute_values = data[attribute].value_counts(normalize=True)
    conditional_gini = 0
    for value in attribute_values.index:
        subset = data[data[attribute] == value]
        subset_gini = calculate_gini_index(subset, target_column)
        conditional_gini += attribute_values[value] * subset_gini
    return conditional_gini

def calculate_gini_gain(data, attribute, target_column):
    total_gini = calculate_gini_index(data, target_column)
    attribute_gini = calculate_gini_attribute(data, attribute, target_column)
    gini_gain = total_gini - attribute_gini
    return gini_gain

# Atributos e coluna alvo
attributes = ["age", "income", "student", "credit_rating"]
target_column = "Class: buys_computer"

# Cálculo do ganho de informação para cada atributo
information_gain_results = {attr: calculate_information_gain(df, attr, target_column) for attr in attributes}
sorted_information_gain = sorted(information_gain_results.items(), key=lambda item: item[1], reverse=True)

# Cálculo do ganho de Gini para cada atributo
gini_gain_results = {attr: calculate_gini_gain(df, attr, target_column) for attr in attributes}
sorted_gini_gain = sorted(gini_gain_results.items(), key=lambda item: item[1], reverse=True)

print(sorted_information_gain, sorted_gini_gain)
import pandas as pd
import numpy as np

df = pd.read_csv('/home/ufc/savim/machine-learning-ufc/AV1/buys_computer_data.csv')

df = df.drop(['RID'],axis=1)

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = (group.count(class_val) / size)
            score += proportion * proportion
        gini += (1.0 - score) * (size / n_instances)
    return gini

def entropy(groups, classes):
    # Calculate the entropy for a split dataset
    n_instances = float(sum([len(group) for group in groups]))
    entropy = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = (group.count(class_val) / size)
            if proportion > 0:
                score += proportion * np.log2(proportion)
        entropy -= (score / np.log2(len(classes))) * (size / n_instances)
    return entropy

def split_groups(df, column):
    unique_values = df[column].unique()
    groups = []
    for value in unique_values:
        group = df[df[column] == value]['Class: buys_computer'].tolist()
        groups.append(group)
    return groups

target_column = 'Class: buys_computer'
attributes = df.columns.drop(target_column)
classes = df[target_column].unique()

gini_scores = {}
entropy_scores = {}

for attribute in attributes:
    groups = split_groups(df, attribute)
    gini_scores[attribute] = gini_index(groups, classes)
    entropy_scores[attribute] = entropy(groups, classes)

sorted_gini = sorted(gini_scores.items(), key=lambda item: item[1])
sorted_entropy = sorted(entropy_scores.items(), key=lambda item: item[1])

print(f'gini: {sorted_gini}')
print(f'entropy: {sorted_entropy}')