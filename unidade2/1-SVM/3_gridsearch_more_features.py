from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Carregar o dataset de diabetes
diabetes_df = load_diabetes_clean_dataset()

# Separando X (features) e y (rótulos)
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values

# Dividindo os dados em treino e teste com stratify para manter a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instanciar o classificador SVM com kernel RBF
svm = SVC(kernel='rbf')

# Definindo o espaço de busca para os parâmetros C e gamma
parameters = {'C': [0.1, 1, 10], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}

# Instanciar o GridSearchCV com validação cruzada de 5 folds
searcher = GridSearchCV(svm, param_grid=parameters, cv=5)

# Ajustar o GridSearchCV aos dados de treino
searcher.fit(X_train, y_train)

# Reportar os melhores parâmetros e a melhor acurácia de validação cruzada
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)

# Reportar a acurácia no conjunto de teste usando os melhores parâmetros
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
