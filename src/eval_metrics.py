import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
