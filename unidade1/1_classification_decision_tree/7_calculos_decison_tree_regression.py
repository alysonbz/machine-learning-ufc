# PACKAGES -------------------------------------------------------------------------------------------------------------
from src.utils import load_house_price_dataset
import numpy as np
import pandas as pd


# DADOS ----------------------------------------------------------------------------------------------------------------
df_house = load_house_price_dataset()


# EXERCICIO ------------------------------------------------------------------------------------------------------------

# -> Questão 1) Realize os calculos de Count, Average, Standard Deviation e Coeff. of Variation da coluna 'price' do dt
def questao1(coluna):
    agg_func = {
    'Count': 'count',
    'Mean': lambda x: round(x.mean(), 2),
    'Std': lambda x: round(x.std(), 2),
    'CV': (lambda x: round((x.std() / x.mean()) * 100, 2))
    }
    return coluna.agg(agg_func)



# -> Questão 2) Realize os cálculos de S(T,X) para todas as colunas do dt, com exceção da coluna target. Armazene a
# resposta em um df e retorne.

## Deu erro quando utilizado a função std(): a variável 'bathroom' possui apenas uma observação na classe '4', e retornava
## NaN como std. Pesquisei no CHAT e obtive:
    # Se você tiver apenas uma observação em seu conjunto de dados e calcular o desvio padrão em Python,
    # poderá obter NaN(Not-a-Number) como resultado. Este é o comportamento esperado porque o desvio padrão
    # não pode ser calculado com apenas um ponto de dados, uma vez que mede a dispersão ou dispersão de um
    # conjunto de dados e requer pelo menos dois pontos de dados para ser calculado.
## Implementei uma função customizada da std(), que retornará zero caso a variável possua apenas 1 observação.
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


# -> Questão 3) Realize o cálculo de SDR(T,X) para todas as colunas do dataset, excessão da coluna target. Armazene a
# resposta em um df e retorne o id do atributo que possui o maior SDR.
def questao3(tab_estatisticas, tab_S):
    tab_SDR =  pd.DataFrame({'variavel': tab_S['variavel'], 'SDR': tab_estatisticas['Std'] - tab_S['S']})
    maior_SDR = tab_SDR[tab_SDR['SDR'] == tab_SDR['SDR'].max()]['variavel'].values[0]
    return tab_SDR, maior_SDR


# ----------------------------------------------------------------------------------------------------------------------

## -> Q1)
tab_estatisticas = questao1(df_house['price'])

print('Questão 1 -----------------------\n',
      tab_estatisticas, '\n')

## -> Q2)
tab_S = questao2(df_house)

print("Questão 2 -----------------------\n",
      tab_S, '\n')

## -> Q3)
tab_SDR, maior_SDR = questao3(tab_estatisticas, tab_S)

print("Questão 3 -----------------------\n",
      tab_SDR,
      '\n\nMaior SDR: ', maior_SDR)

