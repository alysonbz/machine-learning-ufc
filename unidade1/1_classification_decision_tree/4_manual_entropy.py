import math
import pandas as pd

#Sugestão de Fluxo

# construir dataset usando pandas - dataset do slide 11
exam_result = ["Pass",
             "Fail",
             "Fail",
             "Pass",
             "Fail",
             "Fail",
             "Pass",
             "Pass",
             "Pass",
             "Pass",
             "Pass",
             "Pass",
             "Fail",
             "Fail",
             "Fail"]
other_online_curses = ["Y",
                       "N",
                       "y",
                       "Y",
                       "N",
                       "Y",
                       "Y",
                       "Y",
                       "n",
                       "n",
                       "y",
                       "n",
                       "y",
                       "n",
                       "n"]
student_background = ["Maths",
                      "Maths",
                      "Maths",
                      "CS",
                      "Other",
                      "Other",
                      "Maths",
                      "CS",
                      "Math",
                      "CS",
                      "CS",
                      "Maths",
                      "Other",
                      "Other",
                      "Maths"]
working_status = ["NW",
                  "W",
                  "W",
                  "NW",
                  "W",
                  "W",
                  "NW",
                  "NW",
                  "W",
                  "W",
                  "W",
                  "NW",
                  "W",
                  "NW",
                  "W"]

aprovacao_alunos = pd.DataFrame({'Exam Result': exam_result, 'Other Oline Curses': other_online_curses, 'Student Background': student_background, 'Working Status': working_status})

'''aprovacao_alunos[['Exam Result', 'Other Oline Curses', 'Student Background', 'Working Status']] = aprovacao_alunos[['Exam Result', 'Other Oline Curses', 'Student Background', 'Working Status']].astype(int)'''
print(aprovacao_alunos)

# passar dataset para função tab1 e calcular a entropia de cada coluna
def tab1(colunas):
    valores_colunas = colunas.value_counts()
    total_valores_colunas = len(colunas)
    entropia = 0
    for valores in valores_colunas:
        probabilidade = valores / total_valores_colunas
        entropia += probabilidade * math.log2(probabilidade)
    entropia *= -1
    return entropia

resultados_entropia = {}
for colunas in aprovacao_alunos.columns:
    entropia = tab1(aprovacao_alunos[colunas])
    resultados_entropia[colunas] = entropia
    print(f'Entropia de {colunas}: {entropia}')

print()


def tab2():
    print(None)


tab1()

tab2()

#Sugestão de fluxo
# 1- construir o dataset usando pandas - Dataset do slide 11

# 2 - passar o dataset para função tab1 e calcular a entropia de cada coluna

# 3 - Armazenar os valores calculados em uma estrutura  de dados e exibir a tabela de entropia

# 4 - implementar a tab com resultados da tab1 para definição do novo nó pai e calcular novamente  a entropia para tab2
# Exibir tab2 va print.
