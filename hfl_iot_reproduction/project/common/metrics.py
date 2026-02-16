import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def onehot(y, nclass):
    import numpy as np
    y = np.asarray(y).astype(int) - 1  # HAR labels 1..6
    Y = np.zeros((y.shape[0], nclass), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y
