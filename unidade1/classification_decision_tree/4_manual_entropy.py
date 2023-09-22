import pandas as pd
import numpy as np

## DADOS

df = pd.DataFrame({
    'Exam Result': ['Pass', 'Fail', 'Fail', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass',
                    'Fail', 'Fail', 'Fail'],
    'Other online courses': ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N'],
    'Student background': ['Maths', 'Maths', 'Maths', 'CS', 'Other', 'Other', 'Maths', 'CS', 'Maths', 'CS', 'CS',
                           'Maths', 'Other', 'Other', 'Maths'],
    'Working Status': ['NW', 'W', 'W', 'NW', 'W', 'W', 'NW', 'NW', 'W', 'W', 'W', 'NW', 'W', 'NW', 'W']
})

## FUNÇÕES

def calculate_information_gain(data, split_feature, target_feature='Exam Result'):
    parent_entropy = entropy(data[target_feature])
    children = []
    labels = []

    for value in data[split_feature].unique():
        subset = data[data[split_feature] == value][target_feature]
        children.append(subset)
        labels.append(f"{split_feature}={value}")

    avg_entropy = average_entropy(children)
    gain = parent_entropy - avg_entropy
    return {
        'Feature': split_feature,
        'Entropy': parent_entropy,
        'Average Entropy': avg_entropy,
        'Information Gain': gain,
        'Child Nodes': labels
    }

def entropy(var):
    unique_values, counts = np.unique(var, return_counts=True)
    prob_list = counts / len(var)
    entropy = -np.sum(prob_list * np.log2(prob_list))
    return entropy

def average_entropy(children):
    total = sum(len(node) for node in children)
    return sum([(len(node) / total) * entropy(node) for node in children])

## CÁLCULO DE GANHO DE INFORMAÇÃO PARA CADA CARACTERÍSTICA

results = []
features_to_split = ['Working Status', 'Student background', 'Other online courses']

for feature in features_to_split:
    result = calculate_information_gain(df, feature)
    results.append(result)

results_df = pd.DataFrame(results)

print(results_df)
