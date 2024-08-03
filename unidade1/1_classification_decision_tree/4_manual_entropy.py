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


'''def tab1(colunas):
    valores_coluna = colunas.value_counts()
    total_valores_coluna = len(colunas)

    entropia = 0
    for valores in valores_coluna:
        probabilidade = valores / total_valores_coluna
        entropia -= probabilidade * math.log2(probabilidade)

    return entropia

resultados_entropia = {}
for colunas in aprovacao_alunos.columns:
    entropia = tab1(aprovacao_alunos[colunas])
    resultados_entropia[colunas] = entropia
    print(f'Entropia de {colunas}: {entropia}')

# armazenar valores calculados em uma estrutura de dados e exibir a tabela de entropia
tabela_entropia = pd.DataFrame.from_dict(resultados_entropia, orient='index', columns=['Entropia'])
print(f'Tabela de Entopia:\n{tabela_entropia}')

# implementar tab2 com resultados da tab1 para definição do novo nó pai e calcular novamente a entropia
def tab2(aprovacao_alunos, resultados_entropia):
    no_pai = min(resultados_entropia)
    coluna_removida = aprovacao_alunos.drop(columns=[no_pai])
    entropia_pai = tab1(coluna_removida)
    return no_pai, entropia_pai

resultados_entropia2 = {}
for colunas in aprovacao_alunos.columns:
    entropia2 = tab1(aprovacao_alunos[colunas])
    resultados_entropia2[colunas] = entropia2
    print(f'Entropia de {colunas}: {entropia2}')

#Exibir tab2 va print

no_pai, entropia2 = tab2(aprovacao_alunos, resultados_entropia)
print('\n')
print(f'Novo nó pai: {no_pai}')
print(f'Entropia de {no_pai}: {entropia2}')

aprovacao_alunos_reduzido = aprovacao_alunos.drop(columns=[no_pai])
resultados_entropia2 = {}
for coluna in aprovacao_alunos_reduzido.columns:
    entropia2 = tab1(aprovacao_alunos_reduzido[coluna])
    resultados_entropia2[coluna] = entropia2

tabela_entropia2 = pd.DataFrame.from_dict(resultados_entropia2, orient='index', columns=['Entropia'])
print(f'Segunda Tabela de Entopia:\n{tabela_entropia2}')


'''