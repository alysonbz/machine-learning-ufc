# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import BaggingClassifier

from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


class Bagging:
    def _init_(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.oob_scores = []
        self.classifiers = []
        self.oob_samples = []

    def fit(self, X_train, y_train):
        self.X_train = X_train #armazena os dados de treinamento
        self.y_train = y_train
        n_samples = X_train.shape[0] #numero de amostras do conjunto de treinamento
        self.classifiers = []
        self.oob_samples = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True) #Seleciona amostras com substituição para criar uma amostra bootstrap.
            oob_indices = np.setdiff1d(np.arange(n_samples), indices) #Indica as amostras que não foram incluídas na amostra bootstrap (out-of-bag).

            X_sampled = X_train[indices] #Subconjunto dos dados de treinamento selecionado pelas indices.
            y_sampled = y_train[indices]

            classifier = self.base_estimator._class_() #Cria uma nova instância do modelo base e ajusta-o aos dados amostrados.
            classifier.fit(X_sampled, y_sampled)

            self.classifiers.append(classifier)#Armazena o classificador ajustado.
            self.oob_samples.append((oob_indices, y_train[oob_indices]))#Armazena as amostras OOB.

    def predict(self, X):
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])#Armazena as previsões de cada classificador para o conjunto de dados X.
        #Realiza uma votação majoritária para determinar a classe final de cada amostra:
        #Aplica a função ao longo do eixo especificado (0, neste caso), contando as frequências de cada classe
        #e escolhendo a mais frequente (np.bincount(x).argmax())
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_votes

    def oob_score(self):
        n_samples = self.X_train.shape[0]
        n_classes = np.max(self.y_train) + 1
        predictions = np.zeros((n_samples, n_classes)) #Matriz para armazenar contagens de previsões OOB para cada amostra e classe.
        oob_counts = np.zeros(n_samples) #Contagem de quantas vezes cada amostra foi incluída nas amostras OOB.

        for classifier, (oob_indices, oob_labels) in zip(self.classifiers, self.oob_samples):
            if len(oob_indices) > 0:
                oob_indices = oob_indices.astype(int)  # Garantir que oob_indices seja do tipo inteiro
                oob_predictions = classifier.predict(self.X_train[oob_indices]) #Previsões para as amostras OOB.
                for idx, pred in zip(oob_indices, oob_predictions):
                    predictions[idx, pred] += 1
                    oob_counts[idx] += 1

        oob_mask = oob_counts > 0#Máscara para identificar amostras que foram OOB
        oob_predictions = np.argmax(predictions, axis=1) #Determina a classe com a maior contagem para cada amostra.
        oob_score = np.mean(oob_predictions[oob_mask] == self.y_train[oob_mask]) #Calcula a pontuação OOB como a proporção
        # de previsões corretas entre as amostras OOB.

        return oob_score


# Set seed for reproducibility
SEED=1
df = indian_liver_dataset()
X = df.drop(['is_patient','gender'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)
# Instantiate dt
dt = DecisionTreeClassifier(random_state=1, min_samples_leaf=8)

# Instantiate bc
bc = Bagging(base_estimator=dt, n_estimators=50)

# Fit bc to the training set
bc.fit(X_train ,y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score()

# Evaluate acc_test
acc_test = accuracy_score(y_pred,y_test)

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
