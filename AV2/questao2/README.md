# Tutorial de Execução

Este é um projeto que utiliza modelos de aprendizado de máquina para classificação usando um conjunto de dados de classificaç~ao do kaggle. O projeto contém módulos separados para implementação dos modelos, avaliação e um arquivo principal para execução.

## Requisitos

- Python 3.x instalado
- Pacote scikit-learn instalado (`pip install scikit-learn`)

## Estrutura do Projeto

* AV2/
    *   questao2/
    *
      * best_model.py
      * model_evaluation.py
      * main.py
      * README.md

### Detalhes do Projeto
*   questao2/ - Contém os módulos para implementação dos modelos e avaliação.
* main.py - Arquivo principal para execução do projeto.
* best_model.py - Implementação para selecionar o melhor modelo com base na acurácia.
* models/model_evaluation.py - Este módulo conterá a lógica para avaliar os modelos com base no relatório de classificação e na matrix de confusão.

### Observações:

* O arquivo main.py contém a lógica principal do projeto.
* Os modelos e a lógica de seleção do melhor modelo estão implementados nos módulos dentro da pasta questao2.
O conjunto de dados utilizado é um conjunto sobre classificação de doenças com base nos sintomas
, mas pode ser substituído por outro conjunto de dados de sua preferência