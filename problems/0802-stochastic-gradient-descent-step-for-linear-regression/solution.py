import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.array(weights, dtype=float)   # copy so the caller's array isn't mutated
    n_samples = X.shape[0]

    for t in range(n_iter):
        i = t % n_samples                # cycle through samples in order
        r = X[i] @ w - y[i]              # scalar residual for this one sample
        w = w - 2 * learning_rate * r * X[i]

    return w.tolist()
