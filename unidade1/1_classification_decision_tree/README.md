# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 

### Questão 1

```[1_decision_tree.py](1_decision_tree.py)```

#### Classificação por Árvore de decisão

Nesta questão você vai realizar uma predição com arvore de decisão e gerar a acurácia.
#### Instruções:

1)  Importe `` DecisionTreeClassifier `` .
   
2)  Instancie  ``DecisionTreeClassifier`` com máximo de 6 nós. 

3)  divida o dataset em conjunto para treino e teste com a função train_test_split
4) Execute o comando ``fit`` com os dados de ``X_train`` e ``y_train`` como argumentos.
6) Execute uma predição com a função ``predict`` do ``dt``. Atribua como argumento ``X_test``
7) Print as predições realizadas``y_pred``
8) Print a acurácia do modelo.


### Questão 2

[2_compare_classifcation.py](2_compare_classifcation.py)

#### Comparação de classificações

Nesta questão você vai realizar uma comparação no processo de calssificação

#### Instruções:

1)  Implente a etapa de predição dentro da função  ``process_classifier``. print a acurácia nesta função
   
2)  Instancie a regressão logística e arvore de decisão. 

3) divida o dataset em conjunto para treino e teste com a função train_test_split

4) chame a função de predição  process_classifier para cada objeto


### Questão 3

[3_decision_tree_entropy.py](3_decision_tree_entropy.py)

#### utilização da entropia

Nesta questão você vai realizar uma análise da entropia

#### Instruções:

1)  Implente a etapa de predição dentro da função  ``process_classifier``. print a acurácia nesta função
   
2)  Instancie a regressão logística e arvore de decisão. 

3) divida o dataset em conjunto para treino e teste com a função train_test_split

4) chame a função de predição  process_classifier para cada objeto

### Questão 4

[4_manual_entropy.py](4_manual_entropy.py)

#### utilização da entropia manual

Nesta questão você vai realizar uma análise da entropia manual

#### Instruções:

1)  Implemente a etapa de predição dentro da função  cada tabela de calculo da entropia para arvore de decisão.

### Questão 5

[5_regression_decision_tree.py](5_regression_decision_tree.py)

#### utilização da arvore de decisão para regressão

Nesta questão você vai realizar uma regressão com árvore de decisão

#### Instruções:

1)  importe DecisionTreeRegressor
2) instancie o regressor
3) Aplique a função fit no conjunto de treino
4) Aplique a função predict no conjunto de teste
5) calcule o erro quadrático médio
6) Cacule a raíz quadrada do erro quadrático médio

### Questão 6


[6_linear _regression_vs_decision_tree.py](6_linear%20_regression_vs_decision_tree.py)

#### comparação da regressao linear com a arvore de decisão para regressão

Nesta questão você vai realizar uma regressão com árvore de decisão e regressão linear apra comparação

#### Instruções:

1)  importe DecisionTreeRegressor
2) instancie o regressor
3) Aplique a função fit no conjunto de treino
4) Aplique a função predict no conjunto de teste
5) calcule o erro quadrático médio
6) Cacule a raíz quadrada do erro quadrático médio

### Questão 7

[7_calculos_decison_tree_regression.py](7_calculos_decison_tree_regression.py)

####  arvore de decisão para regressão

Nesta questão você vai realizar os calculos para regressão com árvore de decisão 

#### Instruções:

1) Na função ``questão1`` realize os cáculos de count, average, Standard Deviation e Coeff. of Variation da coluna price do dataset. Retorne os valores.
2) na função ``questao2`` realize os cáulos de S(T,X) para todas as colunas do dataset, com exceção da coluna target. Armazene as respostas em um dataframe e retorne.
3) na função ``questão3`` realize o calculo de SDR(T,X) para todas as colunas do dataset, exceção da coluna target. Amazene a resposta em um dataframe e retorne o id do atributo que possui o maior SDR.



   

