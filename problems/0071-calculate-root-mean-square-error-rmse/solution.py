
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	y_true = np.array(y_true, dtype=float)
	y_pred = np.array(y_pred, dtype=float)

	if (y_true.shape != y_pred.shape) or (y_true.size == 0) or (y_pred.size == 0):
		raise ValueError
		return
		
	rmse_res = np.sqrt((1/y_true.size) * np.sum((y_true - y_pred)**2)) 
	return round(rmse_res,3)
