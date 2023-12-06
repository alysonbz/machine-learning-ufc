import pandas as pd


file_path = 'C:\\Users\\anime\\PycharmProjects\\machine-learning-ufc\\AV2\\adult.csv'
df = pd.read_csv(file_path)


df = df[(df['workclass'] != '?') & (df['occupation'] != '?')]


print("Após a remoção de linhas com '?':")
print(df.info())


new_file_path = 'C:\\Users\\anime\\PycharmProjects\\machine-learning-ufc\\adult_novo.csv'
df.to_csv(new_file_path, index=False)

print(f"O novo CSV foi salvo em: {new_file_path}")

