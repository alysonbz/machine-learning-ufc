import pandas as pd
import numpy as np


# FUNÇÕES --------------------------------------------------------------------------------------------------------------

## ---> Calculo da Entropia
def entropy(var):
    unique_values, counts = np.unique(var, return_counts=True)
    prob_list = counts / len(var)
    entropy = -np.sum(prob_list * np.log2(prob_list))
    return entropy


## ---> Calculo da Entropia Média
def average_entropy(children):
    total = sum(len(node) for node in children)
    return sum([(len(node) / total) * entropy(node) for node in children])


## ---> Cálculo do ganho de informação
def information_gain(parent_entropy, children):
    avg = average_entropy(children)
    return parent_entropy - avg


# TABELA 1 -------------------------------------------------------------------------------------------------------------

def tab1():
    Parent = entropy(df['Exam Result'])
    W = df[df['Working Status'] == 'W']['Exam Result']
    NW = df[df['Working Status'] == 'NW']['Exam Result']
    Maths = df[df['Student background'] == 'Maths']['Exam Result']
    CS = df[df['Student background'] == 'CS']['Exam Result']
    Other = df[df['Student background'] == 'Other']['Exam Result']
    Y = df[df['Other online courses'] == 'Y']['Exam Result']
    N = df[df['Other online courses'] == 'N']['Exam Result']

    node_list = [W, NW, Maths, CS, Other, Y, N]
    avg_ent_list = [[W, NW]] * 2 + [[Maths, CS, Other]] * 3 + [[Y, N]] * 2

    table_data = {
        'node': ['Parent', 'W', 'NW', 'Maths', 'CS', 'Other', 'Y', 'N'],
        'entropy node': [Parent] + [entropy(node) for node in node_list],
        'average entropy': [''] + [average_entropy(nodes) for nodes in avg_ent_list],
        'information gain': [''] + [information_gain(Parent, nodes) for nodes in avg_ent_list]
    }

    return pd.DataFrame(table_data)


# TABELA 2 -------------------------------------------------------------------------------------------------------------

def tab2(Maths):  # Adicione Maths como um argumento
    Parent = entropy(Maths)
    W = df[(df['Working Status'] == 'W') & (df['Student background'] == 'Maths')]['Exam Result']
    NW = df[(df['Working Status'] == 'NW') & (df['Student background'] == 'Maths')]['Exam Result']
    Y = df[(df['Other online courses'] == 'Y') & (df['Student background'] == 'Maths')]['Exam Result']
    N = df[(df['Other online courses'] == 'N') & (df['Student background'] == 'Maths')]['Exam Result']

    node_list = [W, NW, Y, N]
    avg_ent_list = [[W, NW]] * 2 + [[Y, N]] * 2

    table_data = {
        'node': ['Parent', 'W', 'NW', 'Y', 'N'],
        'entropy node': [Parent] + [entropy(node) for node in node_list],
        'average entropy': [''] + [average_entropy(nodes) for nodes in avg_ent_list],
        'information gain': [''] + [information_gain(Parent, nodes) for nodes in avg_ent_list]
    }

    return pd.DataFrame(table_data)


# Exemplo de uso
df = pd.DataFrame({
    'Exam Result': ['Pass', 'Fail', 'Fail', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass',
                    'Fail', 'Fail', 'Fail'],
    'Other online courses': ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N'],
    'Student background': ['Maths', 'Maths', 'Maths', 'CS', 'Other', 'Other', 'Maths', 'CS', 'Maths', 'CS', 'CS',
                           'Maths', 'Other', 'Other', 'Maths'],
    'Working Status': ['NW', 'W', 'W', 'NW', 'W', 'W', 'NW', 'NW', 'W', 'W', 'W', 'NW', 'W', 'NW', 'W']
})

Maths = df[df['Student background'] == 'Maths']['Exam Result']  # Defina a variável Maths
print("Tabela 1:")
print(tab1())

print("\nTabela 2:")
print(tab2(Maths))  # Passe Maths como argumento para a função tab2()
