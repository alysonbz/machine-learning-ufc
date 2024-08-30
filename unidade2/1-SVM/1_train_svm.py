import matplotlib.pyplot as plt
from src.utils import load_wine_dataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

wine = load_wine_dataset()

# Separando os dados em X e y
X = wine.drop(['Quality'], axis=1)
y = wine['Quality'].values

# Dividindo em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo SVM linear
svm = SVC(kernel='linear')

# Ajustando o classificador SVM aos dados de treino
svm.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = svm.predict(X_test)

# Exibindo o relatório de classificação
print(classification_report(y_test, y_pred))

# Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Exibindo a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot()
plt.show()
