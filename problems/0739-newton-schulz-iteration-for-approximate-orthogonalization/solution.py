import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    X = np.array(M, dtype=float)
    X = X / (np.linalg.norm(X) + 1e-7) # Frobenius norm by default

    for _ in range(num_iters):
        A = X @ X.T # Gram matrix (X X^T)
        X = a * X + b * (A @ X) + c * (A @ (A @ X))

    return X.tolist()