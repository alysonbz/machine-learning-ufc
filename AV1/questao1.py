import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

## -------------- DADOS
df = pd.read_csv("data.csv",sep=",")
print("DATAFRAME ORIGINAL:")
print(df)
print("-------------------------------------------------\n")

# Mapeando as variáveis categóricas para numéricas para uso no modelo
df_mapped = df.apply(lambda x: pd.factorize(x)[0])


print("DATAFRAME NUMÉRICO:")
print(df_mapped)
print("-------------------------------------------------\n")

# Separando features e target
X = df_mapped.drop(columns='Loan_Status')
y = df_mapped['Loan_Status']


# Instanciando o classificador da árvore de decisão
tree = DecisionTreeClassifier(random_state=42)

# Treinando o modelo
tree.fit(X, y)

# Obtendo a importância das features usando Gini
feature_importance_gini = tree.feature_importances_

# Criando um DataFrame com a importância das features e seus nomes
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Gini Importance': feature_importance_gini})

# Ordenando as features pelo índice de Gini
feature_importance_df = feature_importance_df.sort_values(by='Gini Importance', ascending=False)

# Exibindo as features e suas importâncias pelo índice de Gini
print("Importância das features pelo índice de Gini:")
print(feature_importance_df)

# Obtendo a importância das features usando Entropia
tree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree_entropy.fit(X, y)
feature_importance_entropy = tree_entropy.feature_importances_

# Criando um DataFrame com a importância das features e seus nomes usando Entropia
feature_importance_entropy_df = pd.DataFrame({'Feature': X.columns, 'Entropy Importance': feature_importance_entropy})
# Ordenando as features pelo índice de Entropia
feature_importance_entropy_df = feature_importance_entropy_df.sort_values(by='Entropy Importance', ascending=False)
print("-------------------------------------------------\n")

# Exibindo as features e suas importâncias pelo índice de Entropia
print("\nImportância das features pela Entropia:")
print(feature_importance_entropy_df)

print("-------------------------------------------------\n")

print("\nMelhores duas possibilidades de nó raiz pela Entropia:")
print(feature_importance_entropy_df['Feature'].head(2).tolist())
print("***************************************************************")
print("Melhores duas possibilidades de nó raiz pelo índice de Gini:")
print(feature_importance_df['Feature'].head(2).tolist())

