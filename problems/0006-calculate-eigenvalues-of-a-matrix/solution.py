import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace = matrix[0][0] + matrix[1][1]
	det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	#use quadratic formula to solve
	quad = math.sqrt(trace**2 - (4*det))
	ans = [(trace + quad)/2, (trace - quad)/2]
	return ans