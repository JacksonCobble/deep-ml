import numpy as np

def cholesky_decomposition(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        # diagonal: L_jj = sqrt(A_jj - sum_k<j L_jk^2)
        pivot = A[j, j] - L[j, :j] @ L[j, :j]
        if pivot <= 0:
            return -1
        L[j, j] = np.sqrt(pivot)
        # below diagonal: L_ij = (A_ij - sum_k<j L_ik L_jk) / L_jj, all i>j at once
        L[j+1:, j] = (A[j+1:, j] - L[j+1:, :j] @ L[j, :j]) / L[j, j]
    return L.tolist()