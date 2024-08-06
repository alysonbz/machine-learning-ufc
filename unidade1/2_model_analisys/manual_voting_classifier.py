# Importações necessárias
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Função para carregar o dataset
from src.utils import indian_liver_dataset

# Classe Voting_classifier
class Voting_classifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator  # Lista de estimadores
        self.models = []

    def fit(self, X_train, y_train):
        self.models = []
        for name, classifier in self.base_estimator:
            model = classifier.fit(X_train, y_train)
            self.models.append((name, model))

    def predict(self, X):
        predictions = np.array([model.predict(X) for name, model in self.models])
        majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predictions)
        return majority_votes

# Define a semente para reprodutibilidade
SEED = 1

# Carrega o dataset e prepara os dados
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instancia os classificadores
lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define a lista de classificadores
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Instancia o Voting_classifier
vc = Voting_classifier(base_estimator=classifiers)

# Treina o Voting_classifier no conjunto de treinamento
vc.fit(X_train, y_train)

# Prediz os rótulos do conjunto de teste
y_pred = vc.predict(X_test)

# Avalia a precisão no conjunto de teste
acc_test = accuracy_score(y_pred, y_test)

# Imprime a precisão do conjunto de teste
print('Test set accuracy: {:.3f}'.format(acc_test))


