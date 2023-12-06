import pandas as pd

# Carregar o dataset
file_path = 'C:\\Users\\anime\\PycharmProjects\\machine-learning-ufc\\AV2\\adult.csv'
df = pd.read_csv(file_path)

# Remover linhas onde 'workclass' ou 'occupation' têm valor '?'
df = df[(df['workclass'] != '?') & (df['occupation'] != '?')]

# Exibir informações sobre o dataset após a remoção
print("Após a remoção de linhas com '?':")
print(df.info())

# Exportar o novo DataFrame para um novo arquivo CSV
new_file_path = 'C:\\Users\\anime\\PycharmProjects\\machine-learning-ufc\\adult_novo.csv'
df.to_csv(new_file_path, index=False)

print(f"O novo CSV foi salvo em: {new_file_path}")

