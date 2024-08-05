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