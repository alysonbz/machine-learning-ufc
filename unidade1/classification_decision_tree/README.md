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


