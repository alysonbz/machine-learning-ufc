import math
import pandas as pd
import numpy as np



# Incializando os dados
import pandas as pd
import numpy as np
import math

# Seu dataframe


dt = {'exam_result':["Pass","Fail","Fail","Pass","Fail","Fail","Pass","Pass","Pass","Pass","Pass","Fail","Fail","Fail"],
        'online_courses' : ["Y","N","Y","Y","N","Y","Y","N","N","Y","N","Y","N","N"],
        "student_background" : ["Maths","Maths","Maths","CS","Other","Other","Maths","Maths","CS","CS","Maths","Other","Other","Other"],
        "working_status" : ["NW","W","W","NW","W","W","NW","W","W","W","NW","W","NW","W"]}

data = pd.DataFrame(dt) # Transformando no formato DataFrame


def calculo_entropia(data, idependente, atributos):
    df = pd.DataFrame(data)

    # Função para calcular a entropia
    def entropia(pass_probability, fail_probability):
        """
        Anotações:

        Calcula a entropia de uma distribuição de probabilidade.

            pass_probability: A probabilidade de sucesso.
            fail_probability: A probabilidade de falha.


            Retorna a entropia
        """
        try:
            entropia = -(pass_probability * math.log2(pass_probability) + fail_probability * math.log2(fail_probability))
        except:
            entropia = 0
        return entropia


    for atributo in atributos:

        valores = set(df[atributo]) # valores únicos da variável preditora


        for valor in valores:

            subset = df[df[atributo] == valor] # dados para o valor único da variável preditora

            exam_results = subset[idependente]


            pass_probability = (exam_results == 'Pass').sum() / len(subset) # probabilidades de passar e reprovar
            fail_probability = 1.0 - pass_probability


            entropy_value = entropia(pass_probability, fail_probability)


            print(f"{valor}: Pass Probability = {pass_probability:.2f}, Fail Probability = {fail_probability:.2f}, Entropia = {entropy_value:.2f}")

"""

### Testes

## Student background attribute
degree_math = entropy(4/7, 3/7)

try:
    degree_cs = entropy(4/4,0)
except:
    degree_cs = 0


try:
    degree_other = entropy(0, 4 / 4)
except:
    degree_other = 0

## Other online course attribute

online_Y = entropy(5/8,3/8)
# online_N = entropy(/7 ,/7)

def tab1():
    print(None)

def tab2():
    print(None)
"""

"""
### Outros testes

def entropia(data, coluna):
    valor, cont = np.unique(data[coluna], return_counts=True)
    entropias = [-p * math.log(p,2) for p in cont / len(data)]

    entropia = np.sum(entropias)

    return entropia

entropy_pai = -((8/10)*math.log(8/15)+((7/15)*math.log(7/15)))

def entropy(number_pass: float,number_fail: float):
    calc = -(number_pass*math.log(number_pass,2)+number_fail*math.log(number_fail,2))
    return calc

def avarage_entropy(number_pass,number_fail,entropy1,entropy2):
    calc = (number_pass*entropy1+number_fail*entropy2)
    return calc


dad_entropy = entropy((8/15),(7/15))

## Working status attribute
working = entropy(3/9, 6/9)
not_working = entropy(5/6, 1/6)

"""
### Teste
atributos = ['online_courses', 'student_background', 'working_status']
idependente = 'exam_result'
print(calculo_entropia(data,idependente,atributos))



# Calcular indice de gini extra