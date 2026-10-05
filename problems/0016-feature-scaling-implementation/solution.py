import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here

	#calculate mean and std of each column
	mean = np.mean(data, axis=0, keepdims=True)
	stdev = np.std(data, axis=0, keepdims=True)

	# formula for standardizing data to mean of 0 stdev 1
	standardized_data = (data - mean) / stdev

	# compute min and max values for each column
	x_min = np.min(data, axis=0, keepdims=True)
	x_max = np.max(data, axis=0, keepdims=True)

	#min max normalization formula
	normalized_data = (data - x_min) / (x_max - x_min)

	return standardized_data.tolist(), normalized_data.tolist()