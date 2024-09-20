from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def pipeline_create(model):
    return Pipeline([('scaler', StandardScaler()), ('classifier', model)])


