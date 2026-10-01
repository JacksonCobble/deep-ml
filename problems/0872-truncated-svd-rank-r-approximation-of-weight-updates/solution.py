import numpy as np

def low_rank_approximation(delta_W: np.ndarray, r: int) -> list:
	"""
	Compute the best rank-r approximation of delta_W via truncated SVD.

	Args:
		delta_W: matrix of shape (m, n)
		r: target rank (1 <= r <= min(m, n))

	Returns:
		The rank-r approximation as a nested Python list of shape (m, n).
	"""
	# Your code here
	#compute SVD
	u, s, vt = np.linalg.svd(delta_W, full_matrices=False)

	#truncate matricies to rank r columns
	u_trunc = u[:, 0:r]
	s_trunc = s[0:r]
	v_trunc = vt.T[:, 0:r]

	return u_trunc @ np.diag(s_trunc) @ v_trunc.T 
	pass