import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    A = np.array(A, dtype=float)
    m, n = A.shape
    rank = 0
    for col in range(n):
        if rank == m:
            break
        pivot = rank + np.argmax(np.abs(A[rank:, col]))
        if abs(A[pivot, col]) < tol:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(rank + 1, m):
            A[r] -= (A[r, col] / A[rank, col]) * A[rank]
        rank += 1
    return rank