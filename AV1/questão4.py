# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Carregando e processando os dados
df = pd.read_csv("db_ajustado.csv")

# Selecionando as colunas mais adequadas
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = df['quality']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Aplicar SMOTE para balancear as classes no conjunto de treinamento
smote = SMOTE(random_state=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Inicializando Decision Tree
dt = DecisionTreeClassifier(random_state=1)
#Inicializando Random Forest
rf = RandomForestClassifier(random_state=1)
#Inicializando AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=1)
#Inicializando Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=1)
#Inicializando HistGB /é como se fosse o SGB para classificação
hgb = HistGradientBoostingClassifier(max_depth=4, random_state=1)

# Definindo a lista de classificadores
classifiers = [('Decision Tree', dt), ('Random Forest', rf), ('Adaboost', ada),('Gradient Boosting', gb),("HistGB para classificação", hgb)]

# Iterando sobre a lista de classificadores que foi definida
for clf_name, clf in classifiers:
    # Fit clf para os dados de treinamento
    clf.fit(X_train, y_train)

    # Predição y_pred
    y_pred = clf.predict(X_test)

    # Calculando acurácia
    accuracy = accuracy_score(y_test, y_pred)

    # Avaliando a precisão do clf no conjunto de teste
    print('{:s} : {:.3f}'.format(clf_name, accuracy))
