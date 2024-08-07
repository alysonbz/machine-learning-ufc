import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from collections import Counter


# Dados
data = {
    "credit_history": ["medium", "medium", "high", "low", "low", "low", "high", "medium", "medium", "low", "medium", "high", "high", "low"],
    "salary": ["high", "high", "high", "medium", "low", "low", "low", "medium", "low", "medium", "medium", "medium", "high", "medium"],
    "property": ["no", "yes", "no", "no", "no", "yes", "yes", "no", "no", "no", "yes", "yes", "no", "yes"],
    "loan_status": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
}

# Criar o DataFrame
df = pd.DataFrame(data)

# Mostrar o DataFrame
print(df)

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
        subset = df[df[feature] == value]['loan_status']
        subset_count = len(subset)

        gini_total += (subset_count / total_count) * gini_impurity(subset)
        entropy_total += (subset_count / total_count) * entropy(subset)

    return gini_total, entropy_total


# Calcular o índice de Gini e a entropia para cada feature
features = ['credit_history','salary','property']
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