import pandas as pd
from collections import Counter
import numpy as np

# Dados formatados
dados = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'D.C.', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas'],
    'Median Income': [49936, 68734, 62283, 49781, 70489, 73034, 72812, 85750, 65012, 54644, 55821, 80108, 58728, 70145, 59892, 68718, 63938],
    "Bachelor's Degree or Higher": [0.25, 0.29, 0.28, 0.22, 0.33, 0.39, 0.38, 0.57, 0.31, 0.29, 0.30, 0.32, 0.27, 0.33, 0.25, 0.28, 0.32],
    'White': [0.66, 0.60, 0.54, 0.72, 0.37, 0.68, 0.66, 0.37, 0.62, 0.53, 0.52, 0.21, 0.82, 0.61, 0.79, 0.86, 0.76],
    'Political Leaning': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    'Income': [False, True, False, False, True, True, True, True, True, False, False, True, False, True, False, True, True],
    'Education': [False, False, False, False, True, True, True, True, True, False, False, True, False, True, False, False, True],
    'Diversity': [True, True, True, False, True, True, True, True, True, True, True, True, False, True, False, False, False]
}

# Criar DataFrame
df = pd.DataFrame(dados)

def gini_impurity(labels):
    """Calculate the Gini Impurity for a list of labels."""
    total_count = len(labels)
    if total_count == 0:
        return 0
    label_counts = Counter(labels)
    impurity = 1 - sum((count / total_count) ** 2 for count in label_counts.values())
    return impurity

def entropy(labels):
    """Calculate the entropy for a list of labels."""
    total_count = len(labels)
    if total_count == 0:
        return 0
    label_counts = Counter(labels)
    entropy_value = -sum(
        (count / total_count) * np.log2(count / total_count) for count in label_counts.values() if count != 0)
    return entropy_value

def calculate_gini_and_entropy_for_feature(feature):
    """Calculate the Gini impurity and entropy for a given feature."""
    unique_values = df[feature].unique()
    gini_total = 0
    entropy_total = 0
    total_count = len(df)

    for value in unique_values:
        subset = df[df[feature] == value]['Diversity']
        subset_count = len(subset)

        gini_total += (subset_count / total_count) * gini_impurity(subset)
        entropy_total += (subset_count / total_count) * entropy(subset)

    return gini_total, entropy_total

# Calcular o índice de Gini e a entropia para cada feature
features = ['Median Income', "Bachelor's Degree or Higher", 'White', 'Political Leaning', 'Income', 'Education']
gini_values = {}
entropy_values = {}

for feature in features:
    gini, ent = calculate_gini_and_entropy_for_feature(feature)
    gini_values[feature] = gini
    entropy_values[feature] = ent

# Encontrar as duas melhores features com base no índice de Gini e entropia
best_gini_features = sorted(gini_values, key=gini_values.get)[:2]
best_entropy_features = sorted(entropy_values, key=entropy_values.get)[:2]

print("Melhores features pelo índice de Gini:", best_gini_features)
print("Melhores features pela entropia:", best_entropy_features)
