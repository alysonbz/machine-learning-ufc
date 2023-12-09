import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Dados ----------------------------------------------------------------------------------------------------------------

df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/Skyserver.csv')


# Encoding -------------------------------------------------------------------------------------------------------------

# print('Valores unicos em "Class" antes do encoding: ', df['class'].unique())

encoder = preprocessing.LabelEncoder()
df['class'] = encoder.fit_transform(df["class"])

# print('Valores unicos em "Class" depois do encoding: ', df['class'].unique())


# Retirando NAs --------------------------------------------------------------------------------------------------------

df.dropna(inplace=True)


# Separando alvo e preditores ------------------------------------------------------------------------------------------

X = df.drop('class', axis=1)
y = df['class']


# Padronizando ---------------------------------------------------------------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Dividindo o dados em conjuntos de treinamento e teste ----------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modelos escolhidos e seus respectivos parametros ---------------------------------------------------------------------

models = [(DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'],
                                      'max_depth': [10, 20, 30]}),
          (RandomForestClassifier(), {'criterion': ['gini', 'entropy'],
                                      'max_depth': [10, 20, 30]}),
          (AdaBoostClassifier(), {'n_estimators': [50, 100, 200],
                                  'learning_rate': [0.01, 0.1, 1]}),
          (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200],
                                          'learning_rate': [0.01, 0.1, 1]}),
          (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200],
                                          'learning_rate': [0.01, 0.1, 1],
                                          'subsample': [0.7, 0.8, 0.9]}),
          (SVC(), {'C': [0.01, 0.1, 1, 10]})]


# Loop para treinamento dos modelos ------------------------------------------------------------------------------------

for model, param_grid in models:
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    model_name = model.__class__.__name__
    print(f"Melhores parâmetros para {model_name}:", best_params)
