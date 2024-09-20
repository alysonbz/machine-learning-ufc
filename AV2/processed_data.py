import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

#Leitura dos Dados
data = pd.read_csv('/machine-learning-ufc/AV1/Domestic violence.csv')
print(data.head())

df = data.drop('SL. No', axis=1)


# Verificar qualidade dos dados através da presença de nulos
nAn = df.isna().sum()
string_columns = df.select_dtypes(include='object').columns
unique_values = {column: df[column].unique() for column in string_columns}

#Verificar valores unicos
for column, values in unique_values.items():
    print(f"Valores únicos na coluna '{column}': {values}")

# Codificando as variáveis
le = LabelEncoder()
for column in string_columns:
    df[column] = le.fit_transform(df[column])

scaler = MinMaxScaler(feature_range=(0,1))
scaler_columns = ['Age']  # Se quiser normalizar mais colunas, adicione-as nesta lista
df[scaler_columns] = scaler.fit_transform(df[scaler_columns])

df.to_csv('df_final.csv', index_label=True)