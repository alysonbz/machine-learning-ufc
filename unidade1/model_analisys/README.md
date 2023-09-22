# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 1  

### Questão 1

```[multiples_models.py](multiples_models.py)```

#### testes com multiplos modelos

Nesta questão você vai realizar uma predição com arvore de decisão, knn, regressão logistica e voting classifier e gerar a acurácia.
#### Instruções:

1)  Importe os módulos necessários .
   
2)  Instancie  ``LogisticRegression``.

3)  Instancie  ``KNN``.
4) Instancie  ``DecisionTree``.
6) Itere no loop for o treinamento de cada classificador e print a acurácia de cada um no banco de teste
7) Instancie  ``VotingClassifier``.
8) execute o fit para votingClassifier
9) Obtenha a acurácia do modelo


### Questão 2

[bagging.py](bagging.py)

#### testes com unico modelo, mas várias variações

Nesta questão você vai realizar uma predição com arvore de decisão usando bagging classifier.

#### Instruções:

1)  Importe os módulos necessários .
   
2)  Instancie  ``DecisionTree`` com random_state igual a SEED.

3)  Instancie  ``BaggingClassifier`` para a árvore de decisão com 50 estimadores e random_state igaul a SEED.
4)  Execute a função Fit de bc para o conjunto de treino. 
6)  realize predições e armazene em y_pred
7)   Calcule a acurácia


### Questão 3

[oob_score.py](oob_score.py)

#### testes com modelos mutiplos de uma unica técnica

Nesta questão você vai realizar uma predição com arvore de decisão usando bagging classifier.

#### Instruções:

1)  Importe os módulos necessários .
   
2)  Instancie  ``DecisionTree`` com random_state igual a SEED.

3)  Instancie  ``BaggingClassifier`` para a árvore de decisão com 50 estimadores e random_state igaul a SEED. Defina oob_score True e 50 estimadores.
4)  Execute a função Fit de bc para o conjunto de treino. 
6)  realize predições e armazene em y_pred
7)  Calcule a acurácia oob
8) Calcule a acurácia no banco de teste



### Atividade para análise de modelos

[Atividade_analise_modelos.py](Atividade_analise_modelos.py)


#### Análise de erro e multiplos modelos


#### Instruções:

1)  Implemente uma função que calcula bias e variancia de um modelo de classificação com arvore de decisão. Escolha algum dataset disponivel neste repositório para calcular.
   
2)  Plot o comportamento gráfico trade-off de bias e variância para um modelo de classsificação por arvore de decisão. Prponha as duas listas bias e variance e estime valores em que quando aplicado na equação do erro gere o gráfico trade-off.


### Atividade para implemtação manual

[manual_bagging.py](manual_bagging.py)

[manual_voting_classifier.py](manual_voting_classifier.py)

#### Instruções:

1)  Faça uma implemtação manual do algoritmo bagging. Conforme indicação no código.
   
2)  Faça uma implementação manual do voting_classifier. Considere hard voting como como o caso default de escolha.
