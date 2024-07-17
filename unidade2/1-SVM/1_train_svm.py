import matplotlib.pyplot as plt
from src.utils import load_wine_dataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

wine = load_wine_dataset()

X = wine.drop(['Quality'],axis=1)
y = wine['Quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a linear SVM
svm = SVC(gamma=2)

# fit svm classifier
svm.fit(X_train, y_train)

# predict with svm classifier
y_pred = svm.predict(X_test)

# print classification report matrix
print (classification_report())

# show confusion matrix
cm = ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot()
plt.show()
