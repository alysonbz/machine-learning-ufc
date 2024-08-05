import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,  classification_report, mean_squared_error as MSE, roc_curve, auc, RocCurveDisplay

df = pd.read_csv("/home/ufc/savim/machine-learning-ufc/AV1/plant_growth_data.csv")

#print(df.describe())

# Criação de um LabelEncoder para cada coluna categórica
le_soil = LabelEncoder()
le_water = LabelEncoder()
le_fertilizer = LabelEncoder()

# Aplicação do LabelEncoder nas colunas categóricas
df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])
df['Water_Frequency'] = le_water.fit_transform(df['Water_Frequency'])
df['Fertilizer_Type'] = le_fertilizer.fit_transform(df['Fertilizer_Type'])

X = df.drop(['Growth_Milestone'],axis=1).values
y = df[['Growth_Milestone']].values

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X)

# Instanciação de um DecisionTreeClassifier 'dt' com um máximo de 6 níveis de profundidade
dt = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=1, min_samples_split=2, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y,random_state=1)

# Fit dt to the training set
dt.fit(X_train,y_train)

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_test,y_pred)

#print the score
print("Test set accuracy: {:.2f}".format(acc))
#print Classification Report
print(classification_report(y_test, y_pred))

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test**(1/2)

print('Test set RMSE of dt: {:.3f}'.format(rmse_test))

# Curva ROC e AUC
y_prob = dt.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=dt.classes_[1])
roc_auc = auc(fpr, tpr)

# Exibição da curva ROC
plt.figure(figsize=(10, 6))
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Decision Tree')
roc_display.plot()
plt.show()


df.to_csv('C:\\Users\\UFC\\Desktop\\savim\\machine-learning-ufc\\AV1\\plant_growth_data_pos.csv', index=False)
