import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	# get our eigenvalues
	_, s, _ = np.linalg.svd(delta_W)
	rank_norms = [0]
	tot = 0
	for val in s:
		tot += val**2
		rank_norms.append(rank_norms[-1] + val**2)
	
	for idx in range(len(rank_norms)):
		if rank_norms[idx]/tot >= energy_threshold:
			return idx
	
	return 0
	