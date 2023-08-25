from sklearn.impute import SimpleImputer
from src.utils import load_breast_cancer_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def process_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

# Carregue o conjunto de dados
df_breast = load_breast_cancer_dataset()

# Separe os dados em X (características) e y (rótulos)
X = df_breast.values
y = df_breast[['diagnosis']].values.ravel()

# Divida os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Instancie os modelos de regressão logística e árvore de decisão
logreg = LogisticRegression()
dt = DecisionTreeClassifier()

# Instancie o imputador
imputer = SimpleImputer(strategy='mean')

# Ajuste o imputador aos dados de treinamento e transforme ambos os conjuntos de dados
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Chame a função para processar o modelo de regressão logística
process_classifier(logreg, X_train_imputed, X_test_imputed, y_train, y_test)

# Chame a função para processar o modelo de árvore de decisão
process_classifier(dt, X_train_imputed, X_test_imputed, y_train, y_test)
