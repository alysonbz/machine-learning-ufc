import pandas as pd
import math
df = pd.read_csv('Planilha sem título - Página1.csv')

print(df.head)
def tab1():
    print(None)

def tab2():
    print(None)




import numpy as np

import math

# Contar o número de ocorrências de cada valor na variável alvo
value_counts = df['Target variable'].value_counts()

# Calcular a probabilidade de cada valor ocorrer
total_samples = len(df)
P_pass = value_counts['Pass'] / total_samples
P_fail = value_counts['Fail'] / total_samples

# Calcular a entropia do nó "Parent"
entropy_parent = -(P_pass * math.log2(P_pass) + P_fail * math.log2(P_fail))

print('Entropia do nó "Parent":', entropy_parent)


# Dividir o conjunto de dados com base no status de trabalho
working_df = df[df['Working Status'] == 'W']
not_working_df = df[df['Working Status'] == 'NW']

''''# Calcular a entropia para cada subconjunto
entropy_working = -(len(working_df) / total_samples) * math.log2(len(working_df) / total_samples)
entropy_not_working = -(len(not_working_df) / total_samples) * math.log2(len(not_working_df) / total_samples)

print('Entropia do nó "Working":', entropy_working)
print('Entropia do nó "Not Working":', entropy_not_working)
'''

