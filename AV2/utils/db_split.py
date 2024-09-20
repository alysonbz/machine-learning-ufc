from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def split_balance(df, target_column, test_size=0.2, random_state=1):
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    sm = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test




