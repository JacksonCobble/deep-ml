import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X, dtype=float)
	y = np.array(y, dtype=float)

	#implement the normal equation for finding weights of least squares formula
	theta = (np.linalg.inv(X.T @ X)) @ X.T @ y

	#round to 4 decimal places
	return np.round(theta, decimals=4)