# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 1  

### Questão 1

[adaboost.py](1_adaboost.py)

#### Adaboost

Nesta questão você vai realizar uma classificação com Adaboost

#### Instruções:

1)  Importe os módulos necessários .
2) Instancie  uma arvore de decisão com profundidade 2 e random state 1.
3) instancie o AdaBoost com 180 estimadores e random state 1
4) Compute as probabilidade das classes
5) Execute  roc_auc_score no conjunto de teste


### Questão 2

[2_gradient_boost.py](2_gradient_boost.py)

#### Adaboost

Nesta questão você vai realizar uma regressão com gradientBoost

#### Instruções:

1)  Importe os módulos necessários .
2) Instancie  o graiente boost com n_estimators=200, max_depth=4 e  random_state=2.
3) Execute Fit no conjunto de treino
4) Realize as predições no conjunto de teste
5) Calcule MSE
6) Calcule RMSE
7) print RMSE



### Questão 3

[3_SGB.py](3_SGB.py)

#### Adaboost

Nesta questão você vai realizar uma regressão com gradientBoost estocástico.
 
#### Instruções:

1)  Importe os módulos necessários .
2) Instancie  o graiente boost com max_depth=4,  subsample=0.9, max_features=0.75, n_estimators=200 e random_state=2.
3) Execute Fit no conjunto de treino
4) Realize as predições no conjunto de teste
5) Calcule MSE
6) Calcule RMSE
7) print RMSE
