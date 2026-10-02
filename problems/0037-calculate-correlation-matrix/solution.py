import numpy as np

def corr_pair(X, Y):
	# deviations
	dx = X - np.mean(X)
	dy = Y - np.mean(Y)
	# calculate correlation
	# covariance equation, divide by n-1
	cov = np.sum(dx*dy) / (len(X) - 1) 
	return cov / (np.std(X, ddof=1) * np.std(Y, ddof=1))

	bottom = np.sqrt(np.sum)
def calculate_correlation_matrix(X, Y=None):
	# Your code here
	# check optional ys
	if Y is None:
		Y = X 

	#size of cov matrix is (#cols x, #cols y)
	result = np.zeros((X.shape[1], Y.shape[1]))

	#loop over every possible combinationof columns
	for i in range(X.shape[1]):
		for j in range(Y.shape[1]):
			# calculate correlation of position i,j
			result[i,j] = corr_pair(X[:,i], Y[:,j])

	return result

	pass