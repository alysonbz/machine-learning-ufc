import numpy as np
import pandas as pd

from src.utils import load_house_price_dataset

df_house = load_house_price_dataset()


def questao1(coluna):
    agg_func = {
    'Count': 'count',
    'Mean': lambda x: round(x.mean(), 2),
    'SD': lambda x: round(x.std(), 2),
    'CV': (lambda x: round((x.std() / x.mean()) * 100, 2))
    }
    return coluna.agg(agg_func)


# Se você tiver apenas uma observação em seu conjunto de dados e calcular o desvio padrão em Python,
# poderá obter NaN(Not-a-Number) como resultado. Este é o comportamento esperado porque o desvio padrão
# não pode ser calculado com apenas um ponto de dados, uma vez que mede a dispersão ou dispersão de um
# conjunto de dados e requer pelo menos dois pontos de dados para ser calculado.
def custom_std(x):
    if len(x) < 2:
        return 0
    return np.std(x)

def questao2(df):
    colunas = df.columns.difference(['price'])
    var = []
    S = []

    for pred in colunas:
        tabelaS = df_house[pred].value_counts().reset_index()
        tabelaS['std'] = df_house.groupby(pred)['price'].agg(std=custom_std).reset_index()['std']
        tabelaS['P'] = tabelaS['count'] / sum(tabelaS['count'])
        tabelaS['S'] = tabelaS['P'] * tabelaS['std']

        var.append(pred)
        S.append(sum(tabelaS['S']))

    return pd.DataFrame({'variavel': var, 'S': [round(valor, 2) for valor in S]})

def questao3():
    return None


print('Questão 1:\n', questao1(df_house['price']))

print("Questao2:\n",questao2(df_house))
#print("Questao3: ",questao3())


