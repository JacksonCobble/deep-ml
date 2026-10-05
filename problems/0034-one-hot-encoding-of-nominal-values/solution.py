import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	labels = np.array(x)

	if n_col:
		n_classes = n_col
	else:
		n_classes = labels.max() + 1
	
	return np.eye(n_classes)[labels]