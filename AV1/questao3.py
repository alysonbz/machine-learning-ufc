# PACKAGES -------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# FUNÇÃO ---------------------------------------------------------------------------------------------------------------
class RandomClassifier:
    def __init__(self, classifier_type, random_seed=None):
        self.classifier_type = classifier_type
        self.random_seed = random_seed
        self.classifier = None

    def fit(self, X, y):
        classifiers = {
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier()
        }

        self.classifier = classifiers.get(self.classifier_type)

        if self.random_seed:
            np.random.seed(self.random_seed)

        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)


# TESTE DA FUNÇÃO ------------------------------------------------------------------------------------------------------

# Dados
df = pd.read_csv('C:/Users/Thays Ferreira/Downloads/Skyserver.csv')

# Encoding
encoder = preprocessing.LabelEncoder()
df['class'] = encoder.fit_transform(df["class"])

# Retirando NAs
df.dropna(inplace=True)

# Separando alvo e preditores
X = df.drop('class', axis=1).values
y = df['class'].values

# Padronizando
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separando conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=12)

# Treinando os modelos
for classifier in ['knn', 'decision_tree']:
    rc = RandomClassifier(classifier_type=classifier, random_seed=12)
    rc.fit(X_train, y_train)
    y_pred = rc.predict(X_test)
    print('Acurácia do', classifier, 'classifier: ', accuracy_score(y_test, y_pred))

