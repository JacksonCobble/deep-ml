import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# vectors[i] = all observations of feature i  (rows = features)
	n_features = len(vectors)
	n_obs = len(vectors[0])

	# 1. mean of each feature
	means = [sum(row) / n_obs for row in vectors]

	# 2. build the matrix: cov[i][j] = cov(feature i, feature j)
	cov = [[0.0] * n_features for _ in range(n_features)]
	for i in range(n_features):
		for j in range(i, n_features):          # symmetric, so only compute upper triangle
			c = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
					for k in range(n_obs)) / (n_obs - 1)
			cov[i][j] = c
			cov[j][i] = c                        # mirror it
	return cov