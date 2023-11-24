import pandas as pd
from math import log2

# Transformando os dados da imagem para DataFrame
dados = [
    ["Rainy", "Hot", "High", "FALSE", "No"],
    ["Rainy", "Hot", "High", "TRUE", "No"],
    ["Overcast", "Hot", "High", "FALSE", "Yes"],
    ["sunny", "Mild", "High", "FALSE", "Yes"],
    ["sunny", "Cool", "Normal", "FALSE", "Yes"],
    ["sunny", "Cool", "Normal", "TRUE", "No"],
    ["Overcast", "Cool", "Normal", "TRUE", "Yes"],
    ["Rainy", "Mild", "High", "FALSE", "No"],
    ["Rainy", "Cool", "Normal", "FALSE", "Yes"],
    ["sunny", "Mild", "Normal", "FALSE", "Yes"],
    ["Rainy", "Mild", "Normal", "TRUE", "Yes"],
    ["overcast", "Mild", "High", "TRUE", "Yes"],
    ["Overcast", "Hot", "Normal", "FALSE", "Yes"],
    ["Sunny", "Mild", "High", "TRUE", "No"]
]

# Definindo as colunas do DataFrame
colunas = ["Outlook", "Temperature", "Humidity", "Windy", "Play Golf"]

# Criando o DataFrame
df = pd.DataFrame(dados, columns=colunas)
print(df)

# Função para o cálculo de Gini
def calculo_gini(data):
    total_samples = len(data)
    if total_samples == 0:
        return 0.0

    class_counts = data['Play Golf'].value_counts()
    gini = 1.0
    for class_count in class_counts:
        gini -= (class_count / total_samples) ** 2

    return gini

# Função para o cálculo da Entropia
def calculo_entropia(data):
    total_samples = len(data)
    if total_samples == 0:
        return 0.0

    class_probs = data['Play Golf'].value_counts() / total_samples
    entropy = -sum(p * log2(p) for p in class_probs)

    return entropy

# Selecionando variável dependente( rótulo de quem jogou ou nao golfe) e independentes
target_column = 'Play Golf'
features = df.columns.difference([target_column])

# Calculando índice de Gini e Entropia para cada uma das variáveis

gini_scores = {}
entropia_scores = {}

for feature in features:
    data_grouped = df.groupby([feature, target_column]).size().reset_index(name='count')
    gini = calculo_gini(data_grouped)
    entropy = calculo_entropia(data_grouped)
    gini_scores[feature] = gini
    entropia_scores[feature] = entropy

# Obtendo as duas opções de nó raiz (menores valores)
opcao_gini = sorted(gini_scores.items(), key=lambda x: x[1])[:2]
opcao_entropia = sorted(entropia_scores.items(), key=lambda x: x[1])[:2]

print("Índice de Gini para cada característica:")
for feature, gini in opcao_gini:
    print(f"{feature}: {gini}")

print("\nEntropia para cada característica:")
for feature, entropy in opcao_entropia:
    print(f"{feature}: {entropy}")
