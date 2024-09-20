# Tutorial e detalhes

Esse documento detalha um pouco sobre o conteúdo que está dentro de cada um dos scripts.
Para ser objetivo na hora de excutar os arquivos .py, uma recomendação é executar apenas o script "questao2.py", este script
retornará os resultados da primeira questão e da própria segunda questão.

Na pasta documents pode se encontrar:
1) O dataset utilizado.
2) readme com o tutorial.

Na pasta src pode se encontrar os seguintes scripts:
1) Questão 1: Possui o processo do GridSearch para encontrar os melhores parâmetros com todas as especificações
dos valores de cada um dos parâmetros, a inicialização dos modelos e um "for" que executa o pipeline e o GridSearch.
2) Questão: Este script é o **PRINCIPAL**. Ao executar apenas esta questão, será retornado os valores dos melhores parâmetros
encontrados pelo GridSearch para cada modelo e retornará também o que pede em específico na segunda questão: o classification report,
matriz de confusão e o melhor modelo com base na acurácia.


Na pasta utils pode se encontrar os seguintes scripts:
1) calculo_metricas: Este script possui a função Best_model que é solicitada na Questão 2 e também a função 
auxiliar que retorna o classification report e a matriz de confusão para cada um dos modelos.
2) dataset_split: este script possui a função "split()" que divide o dataset em treino e teste.
3) pipeline: Possui uma função chamada meu_pipeline em que são passados os seguintes argumentos: StandardScaler, para
normalizar os dados e o modelo.

