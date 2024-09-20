import matplotlib.pyplot as plt
from src.utils import load_wine_dataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Carrega o dataset
wine = load_wine_dataset()

# Separação das features e do target
X = wine.drop(['Quality'], axis=1)
y = wine['Quality'].values

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treina um SVM linear
svm = SVC(kernel='linear')

# Ajusta o classificador SVM
svm.fit(X_train, y_train)

# Faz a predição com o classificador SVM
y_pred = svm.predict(X_test)

# Exibe o classification report
print(classification_report(y_test, y_pred))

# Mostra a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot()
plt.show()
