from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Carregar o dataset de diabetes
diabetes_df = load_diabetes_clean_dataset()

# Separação das features e do target
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instanciar um SVM com kernel RBF (Radial Basis Function)
svm = SVC(kernel='rbf')

# Instanciar o objeto GridSearchCV e executar a busca pelos melhores parâmetros
parameters = {'C': [0.1, 1, 10], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters, cv=5)

# Ajustar o GridSearchCV com os dados de treino
searcher.fit(X_train, y_train)

# Reportar os melhores parâmetros e o score correspondente
print("Best CV params:", searcher.best_params_)
print("Best CV accuracy:", searcher.best_score_)

# Reportar a acurácia no conjunto de teste usando os melhores parâmetros encontrados
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
