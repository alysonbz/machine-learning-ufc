from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("Domestic violence.csv")

#Remover colunas irrelevantes
df_filtred = df.drop('SL. No', axis=1)

# Verificar qualidade dos dados através da presença de nulos
nAn = df_filtred.isna().sum()
string_columns = df.select_dtypes(include='object').columns
unique_values = {column: df[column].unique() for column in string_columns}

#Verificar valores unicos
for column, values in unique_values.items():
    print(f"Valores únicos na coluna '{column}': {values}")

# Codificando as variáveis
le = LabelEncoder()
for column in string_columns:
    df_filtred[column] = le.fit_transform(df_filtred[column])
# Separando o Dataset
X = df_filtred.drop('Violence ', axis=1)
y = df_filtred['Violence ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

clf = DecisionTreeClassifier()
clf.fit(X_train_res, y_train_res)

y_pred = clf.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
