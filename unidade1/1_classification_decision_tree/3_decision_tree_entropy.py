# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from src.utils import load_breast_cancer_dataset
from sklearn.model_selection import train_test_split
# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

df_breast = load_breast_cancer_dataset()
X = df_breast[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean"]].values
y  = df_breast[['diagnosis']].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test =train_test_split(__, __, __, random_state=42, stratify=y)

# Instancie aas árvores de decisão com 2 possíveis critérios: entropy e gini
dt_entropy = ___(max_depth=8,criterion=___, random_state=42)
dt_gini =___(max_depth=8,criterion=__, random_state=42)

# Fit os objetos dt_gini e dt_entropy
___
___


# Use dt_entropy e dt_gini para realizar predições no conjunto de teste
y_pred_entropy= __
y_pred_gini =

# Evaluate accuracy_entropy
accuracy_entropy = __

accuracy_gini = __

# Print accuracy_entropy
___

# Print accuracy_gini
__