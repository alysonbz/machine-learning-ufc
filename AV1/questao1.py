'''Converta o dataset de imagem para um dataframe e ,utilizando calculo do indice de Gini e
entropia determine as duas possibilidades de nó raíz da árvore de decisão. A ultima coluna
do dataset é a coluna alvo.'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

sick = pd.DataFrame(sick)

colunas = {}
for column in sick.select_dtypes(include=['object']).columns:
    colunas[column] = LabelEncoder()
    sick[column] = colunas[column].fit_transform(sick[column])

print(sick)

X = sick.drop(columns='Drug')
y = sick['Drug']

arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

importancia_features_gini = arvore.feature_importances_
importancia_features_data = pd.DataFrame({'Features':X.columns, 'Importância de Gini': importancia_features_gini})
importancia_features_data = importancia_features_data.sort_values(by='Importância de Gini', ascending=False)

print(f'Importância das features pelo índice de Gini:\n {importancia_features_data}')

arvore_entropia = DecisionTreeClassifier(criterion='entropy', random_state=42)
arvore_entropia.fit(X, y)
importancia_features_entropia = arvore_entropia.feature_importances_

importancia_features_entropia_data = pd.DataFrame({'Features':X.columns, 'Importância da Entropia': importancia_features_entropia})
importancia_features_entropia_data = importancia_features_entropia_data.sort_values(by='Importância da Entropia', ascending=False)

print(f'\nImportância das features pela Entropia: \n{importancia_features_entropia_data}\n')

print(f'\nDuas melhores possibilidades de nó raiz pela Entropia: {importancia_features_entropia_data["Features"].head(2).tolist()}\n')

print(f'\nDuas melhores possibilidades de nó raiz pelo índice de Gini: {importancia_features_data["Features"].head(2).tolist()}\n')