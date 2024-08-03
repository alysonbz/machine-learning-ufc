'''Converta o dataset de imagem para um dataframe e ,utilizando calculo do indice de Gini e
entropia determine as duas possibilidades de nó raíz da árvore de decisão. A ultima coluna
do dataset é a coluna alvo.'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sick = { 'Patient_ID': ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10',
                        'p11','p12','p13','p14'],
         'Age': ['young','young','middle-age','senior','senior','senior',
                 'middle-age','young','young','senior','young','middle-age',
                 'middle-age','senior'],
         'Sex': ['F','F','F','F','M','M','M','F','M','M','M','F','M',
                 'F'],
         'BP': ['high','high','high','normal','low','low','low','normal','low',
                'normal','normal','normal','high','normal'],
         'Cholesterol': ['normal','high','normal','normal','normal','high',
                         'high','normal','normal','normal','high','high',
                         'normal','high'],
         'Drug': ['Drug A','Drug A','Drug B','Drug B','Drug B','Drug A',
                  'Drug B','Drug A','Drug B','Drug B','Drug B','Drug B',
                  'Drug B','Drug A']
}

sick2 = {
    'Age': {'young': 0, 'middle-age': 1, 'senior': 2},
    'Sex': {'F': 0, 'M': 1},
    'BP': {'low': 0, 'normal': 1, 'high': 2},
    'Cholesterol': {'normal': 0, 'high': 1},
    'Drug': {'Drug A': 0, 'Drug B': 1}
}

sick = pd.DataFrame(sick)
sk = sick.replace(sick2)

print(sk)




