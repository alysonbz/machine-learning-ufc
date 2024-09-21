from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Carregar o dataset limpo de diabetes
diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instanciar um SVM com kernel RBF
svm = SVC(kernel='rbf')

# Instanciar o objeto GridSearchCV e realizar a busca
parameters = {'C': [0.1, 1, 10], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters, cv=5, scoring='accuracy')

# Ajustar o GridSearchCV
searcher.fit(X_train, y_train)

# Relatar os melhores parâmetros e a acurácia correspondente
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Relatar a acurácia no conjunto de teste usando os melhores parâmetros
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))