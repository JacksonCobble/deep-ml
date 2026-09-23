import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    num_vars = A.shape[0]        # size of the system — NOT n
    x_old = np.zeros(num_vars)

    for iteration in range(n):   # n = number of iterations, as intended
        x_new = np.zeros(num_vars)
        for j in range(num_vars):
            row_dot = np.dot(A[j], x_old)
            row_dot -= A[j, j] * x_old[j]
            x_new[j] = (b[j] - row_dot) / A[j, j]
        x_old = x_new

    return np.round(x_old, 4).tolist()