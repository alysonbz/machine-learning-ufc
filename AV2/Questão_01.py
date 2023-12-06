import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV


df = pd.read_csv('C:\\Users\\anime\\PycharmProjects\\machine-learning-ufc\\adult_novo.csv')


X = df.drop('income', axis=1)
y = df['income']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'Decision Tree': (DecisionTreeClassifier(), {'classifier__max_depth': [None, 10, 20, 30]}),
    'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 150], 'classifier__max_depth': [None, 10, 20, 30]}),
    'AdaBoost': (AdaBoostClassifier(), {'classifier__n_estimators': [50, 100, 150], 'classifier__learning_rate': [0.01, 0.1, 1.0]}),
    'Gradient Boosting': (
    GradientBoostingClassifier(), {'classifier__n_estimators': [50, 100, 150], 'classifier__learning_rate': [0.01, 0.1, 1.0]}),
    'Stochastic Gradient Boosting': (GradientBoostingClassifier(subsample=0.8, max_features='sqrt'),
                                     {'classifier__n_estimators': [50, 100, 150], 'classifier__learning_rate': [0.01, 0.1, 1.0]}),
    'SVM': (SVC(), {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto'], 'classifier__kernel': ['linear', 'rbf']})
}


best_models = {}

for name, (model, params) in models.items():

    categorical_features = ['workclass', 'occupation']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])


    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
        ])


    model_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


    grid_search = GridSearchCV(model_pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)


    best_models[name] = grid_search.best_estimator_


    print(f'{name}: Melhores Parâmetros - {grid_search.best_params_}')



def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'],
                      yticklabels=['<=50K', '>50K'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Valores Reais')
    plt.show()


def plot_roc_curves(models, X_test, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Curvas ROC para Modelos')

    for (name, model), ax in zip(models.items(), axes.flatten()):
        if hasattr(model.named_steps['classifier'], 'decision_function'):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve((y_test == '>50K').astype(int), y_scores)  # Converte os rótulos para binários
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title(name)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


plot_confusion_matrix(best_models['Decision Tree'], X_test, y_test)


plot_roc_curves(best_models, X_test, y_test)


rf_model = best_models['Random Forest']
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)

plt.figure(figsize=(10, 6))
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Importância')
plt.title('Importância de Características (Permutação) - Random Forest')
plt.show()
