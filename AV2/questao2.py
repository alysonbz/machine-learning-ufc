import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


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


# método para identificar o modelo de maior acurácia -------------------------------------------------------------------

def grid_search_best_model(models, X_train, y_train, X_test, y_test):
    all_results = []

    for classifier_name, classifier, param_grid in models:
        # Grid Search
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        atual_best_model = grid_search.best_estimator_

        # predizendo os valores
        y_pred = atual_best_model.predict(X_test)

        # registrando as informações
        all_results.append({'classifier': classifier_name,
                            'accuracy': accuracy_score(y_test, y_pred),
                            'report': classification_report(y_test, y_pred),
                            'confusion': confusion_matrix(y_test, y_pred)})

    # ordenando os resultados
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)

    best_model = all_results[0]

    return best_model, all_results


# Modelos escolhidos e seus respectivos parametros ---------------------------------------------------------------------

models_params = [('Árvore de Decisão', DecisionTreeClassifier(), {'criterion': ['entropy'],
                                                                  'max_depth': [20]}),
                 ('Random Forest', RandomForestClassifier(), {'criterion': ['gini'],
                                                              'max_depth': [10]}),
                 ('AdaBoost', AdaBoostClassifier(), {'n_estimators': [100],
                                                     'learning_rate': [0.1]}),
                 ('Gardient Boost', GradientBoostingClassifier(), {'n_estimators': [200],
                                                                   'max_depth': [20],
                                                                   'learning_rate': [0.1]}),
                 ('SGB', GradientBoostingClassifier(), {'n_estimators': [100],
                                                        'learning_rate': [0.1],
                                                        'subsample': [0.7]}),
                 ('SVM', SVC(), {'C': [10]})]


# -------------------------------------------------------------------------------------------------------------------- #

best_model, all_results = grid_search_best_model(models_params, X_train, y_train, X_test, y_test)


# classification report
for result in all_results:
    print(result['classifier'])
    print(result['report'])


# matriz de confusão
plt.figure(figsize=(16, 14))
for i, result in enumerate(all_results):
    plt.subplot(2, 3, i+1)
    sns.heatmap(result['confusion'], annot=True, cmap="Blues", cbar=False, fmt="d")
    plt.title(f"\n{result['classifier']}, \nAcurácia: {result['accuracy']}\n")

#plt.show()
