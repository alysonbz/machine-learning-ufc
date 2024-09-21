import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelSelector:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.models = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
            'Gradient Boosting': GradientBoostingClassifier(),
            'SVM': SVC()
        }
        self.best_model = None
        self.best_accuracy = 0

    def fit_and_evaluate(self):
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{model_name} Accuracy: {accuracy:.4f}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model_name

        print(f"\nBest model: {self.best_model} with Accuracy: {self.best_accuracy:.4f}")
        return self.best_model

    def generate_classification_report(self):
        print("\nClassification Report and Confusion Matrix for all models:")
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            print(f"\n{model_name} Classification Report:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))



def main():
    df = pd.read_csv("/home/luissavio/PycharmProjects/machine-learning-ufc/AV2/plant_growth_data_pos.csv")
    X = df.drop(['Growth_Milestone'], axis=1).values
    y = df[['Growth_Milestone']].values.ravel()

    # Inicializando o seletor de modelos
    model_selector = ModelSelector(X, y)

    # Retorna o melhor modelo com base na acurácia
    best_model = model_selector.fit_and_evaluate()

    # Gera o classification report e a matriz de confusão para todos os modelos
    model_selector.generate_classification_report()

# Função que Roda a questão 2 (best_model)
if __name__ == "__main__":
    main()
