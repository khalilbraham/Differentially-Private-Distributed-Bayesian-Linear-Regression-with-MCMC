import numpy as np

def closest_psd(X):
    _, D, E = np.linalg.svd(X)
    v = D > 0
    Y = E[:, v] @ np.diag(D[v]) @ E[:, v].T
    Y = np.real(Y)
    return Y