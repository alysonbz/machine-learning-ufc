# PACKAGES -------------------------------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np  # NOVO
from scipy.stats import mode  # NOVO


# CLASSE BAGGING -------------------------------------------------------------------------------------------------------

class Bagging:
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.classifiers = []  # Lista para armazenar os classificadores
        self.oob_scores = []  # Lista para armazenar os escores OOB

    def fit(self, X, y):
        self.oob_scores = []  # Limpar os escores OOB

        # Loop para criar os classificadores e calcular os escores OOB
        for _ in range(self.n_estimators):

            # Amostragem com reposição -  reduz a variância dos modelos
            indices = np.random.choice(len(X), len(X), replace=True)  # cria amostras de mesma dimensão de X
            X_sample, y_sample = X[indices], y[indices]

            # Cria uma cópia do estimador, treina com as amostras e adiciona o classificador treinado à lista
            classifier = self.base_estimator.__class__()  # __class__() retorna a classe do objeto
            classifier.fit(X_sample, y_sample)
            self.classifiers.append(classifier)

            # Avalia o desempenho do modelo usando as amostras que não foram usadas no treino
            oob_indices = np.setdiff1d(np.arange(len(X)), indices)
            oob_pred = classifier.predict(X[oob_indices])
            oob_acc = accuracy_score(y[oob_indices], oob_pred)
            self.oob_scores.append(oob_acc)

        return self.classifiers

    def predict(self, X):
        predictions = [classifier.predict(X) for classifier in self.classifiers]
        major_votes, _ = mode(predictions, axis=0)
        return major_votes  # Retorna a maioria das previsões

    def oob_score(self):
        return np.mean(self.oob_scores)  # Retorna a média de OOB scores

    def acc_per_classifier(self, X_test, y_test):
        return [accuracy_score(y_test, clf.predict(X_test)) for clf in self.classifiers]  # Calcula a acc de cada classificador


# DADOS ----------------------------------------------------------------------------------------------------------------

# Set seed for reproducibility
SEED = 1
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values


# TESTE ----------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = Bagging(base_estimator=dt, n_estimators=50)

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score()

# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)

print(bc.acc_per_classifier(X_test, y_test))

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
