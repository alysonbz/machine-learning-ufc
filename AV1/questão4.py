import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/home/ufc/savim/machine-learning-ufc/AV1/plant_growth_data_pos.csv")

X = df.drop(['Growth_Milestone'],axis=1).values
y = df[['Growth_Milestone']].values

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X)

# Set seed for reproducibility
SEED=1

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y,random_state=1)

# Instantiate dt - decision tree with min_sample_leaf 0.13 and random state SEED
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Instantiate rf - Random Forest with max_depth=4 and random state SEED
rf = RandomForestClassifier(max_depth=4, random_state=SEED)

# Instantiate ada
ada = AdaBoostClassifier(estimator=dt,  n_estimators=180, random_state=SEED)

# Instantiate gb
gb = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_features=0.75, n_estimators=200,random_state=2)

# Instantiate sgbr
sgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)

# Define the list classifiers
classifiers = [('Decision Tree', dt), ('Random Forest',dt), ('Adaboost', ada), ('Gradient Boosting', gb), ('SGB', sgb)]

# Initialize the plot
plt.figure(figsize=(10, 8))

accuracias = []

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:

    # Fit clf to the training set
    clf.fit(X_train,y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test,y_pred)

    accuracias.append((clf_name, accuracy))

    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{clf_name} (AUC = {roc_auc:.2f})')

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# Plot diagonal line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# Plotar as acurácias em um gráfico de barras
df_accuracies = pd.DataFrame(accuracias, columns=['Model', 'Accuracy'])
colors = ['skyblue', 'lightgreen', 'salmon', 'lightcoral', 'plum']
plt.figure(figsize=(10, 6))
bars = plt.bar(df_accuracies['Model'], df_accuracies['Accuracy'], color=colors)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparação da acurácia dos modelos')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar as porcentagens acima das barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2%}', ha='center', va='bottom')

plt.show()
