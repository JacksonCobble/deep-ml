import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	A = np.array(A, dtype=float)
	m, n = A.shape
	Q = np.zeros((m, n))
	R = np.zeros((n, n))
	for j in range(n):
		v = A[:, j].copy()
		for i in range(j):
			R[i, j] = Q[:, i] @ v        # uses running v (modified GS)
			v -= R[i, j] * Q[:, i]       # strip that direction out
		R[j, j] = np.linalg.norm(v)
		if R[j, j] < 1e-12:
			raise ValueError("columns are linearly dependent")
		Q[:, j] = v / R[j, j]
	return Q, R