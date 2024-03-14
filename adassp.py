import numpy as np


def adassp(X, y, opts):
  """
  This function implements the ADASSP algorithm for regularized least squares.

  Args:
      X: A numpy array of shape (n, d) representing the data matrix.
      y: A numpy array of shape (n,) representing the target vector.
      opts: A dictionary containing options for the algorithm:
          - eps: The epsilon parameter for ADASSP, controlling the accuracy.
          - delta: The delta parameter for ADASSP, controlling the confidence.

  Returns:
      thetahat: A numpy array of shape (d,) representing the estimated parameter vector.
  """

  BX = 1
  BY = 1

  epsilon = opts["eps"]
  delta = opts["delta"]

  n, d = X.shape

  varrho = 0.05

  # Eigenvalue limit
  eta = np.sqrt(d * np.log(6 / delta) * np.log(2 * d**2 / varrho)) * BX**2 / (epsilon / 3)

  XTy = X.T @ y
  XTX = X.T @ X + np.eye(d)

  # SVD and minimum eigenvalue limit
  S, _, _ = np.linalg.svd(XTX)
  logsod = np.log(6 / delta)
  lamb_min = S[-1] + np.random.randn() * BX**2 * np.sqrt(logsod) / (epsilon / 3) - logsod / (epsilon / 3)
  lamb_min = np.maximum(lamb_min, 0)

  # Regularization parameter
  lamb = np.maximum(0, eta - lamb_min)

  # Add noise
  XTyhat = XTy + (np.sqrt(np.log(6 / delta)) / (epsilon / 3)) * BX * BY * np.random.randn(d, 1)
  Z = np.random.randn(d, d)
  Z = 0.5 * (Z + Z.T)
  XTXhat = XTX + (np.sqrt(np.log(6 / delta)) / (epsilon / 3)) * BX**2 * Z

  # Solve regularized linear system
  thetahat = np.linalg.inv(XTXhat + lamb * np.eye(d)) @ XTyhat

  return thetahat