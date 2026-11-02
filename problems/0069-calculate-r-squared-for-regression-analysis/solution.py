
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here 
	y_true = np.array(y_true, dtype=float)
	y_pred = np.array(y_pred, dtype=float)

	# calc sum of squared residuals
	ssr = np.sum((y_true - y_pred)**2)
	# calc total sum of squares
	sst = np.sum((y_true - np.mean(y_true))**2)

	# return r squared value
	return 1 - (ssr/sst)
