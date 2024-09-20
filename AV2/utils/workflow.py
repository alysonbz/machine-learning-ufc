from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def new_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('classifier', model)])


