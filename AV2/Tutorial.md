# Tutorial de Execução do Código

Este tutorial irá guiá-lo passo a passo na execução do código para treinar modelos de aprendizado de máquina e avaliar seus desempenhos com o objetivo de identificar o melhor modelo para o seu conjunto de dados. O projeto inclui scripts para o pré-processamento de dados, ajuste de hiperparâmetros, avaliação de modelos e escolha do melhor modelo.

## 1. Preparar o Ambiente

### 1.1 Instale as Dependências
Certifique-se de ter o Python instalado (recomenda-se a versão 3.7 ou superior). Para instalar as dependências necessárias, execute o seguinte comando em seu terminal ou prompt de comando na raiz do projeto:

```bash
pip install -r requirements.txt
```

### 1.2 Scripts

O script `questao1.py` é responsável por carregar o conjunto de dados, realizar o pré-processamento, ajustar os modelos utilizando GridSearchCV, e identificar os melhores hiperparâmetros para cada modelo.
O script `questao2.py` utiliza os modelos ajustados com os melhores hiperparâmetros e realiza uma avaliação detalhada de cada modelo, escolhendo o que tiver o melhor desempenho.
