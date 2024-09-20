from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def meu_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('classifier', model)])