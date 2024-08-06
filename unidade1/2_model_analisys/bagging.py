from src.utils import indian_liver_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Define a semente para reprodutibilidade
SEED = 1

# Carrega o dataset e prepara os dados
df = indian_liver_dataset()
X = df.drop(['is_patient', 'gender'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['is_patient'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Instancia a árvore de decisão
dt = DecisionTreeClassifier(random_state=SEED)

# Instancia o BaggingClassifier com a árvore de decisão como base
bc = BaggingClassifier(estimator=dt, n_estimators=50, random_state=SEED)

# Treina o BaggingClassifier no conjunto de treinamento
bc.fit(X_train, y_train)

# Prediz os rótulos do conjunto de teste
y_pred = bc.predict(X_test)

# Avalia a precisão no conjunto de teste
acc_test = accuracy_score(y_test, y_pred)

# Imprime a precisão do conjunto de teste
print('Test set accuracy of bc: {:.2f}'.format(acc_test))
