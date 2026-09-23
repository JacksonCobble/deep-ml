import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	mat = np.hstack([np.array(A, dtype=float), np.array(b, dtype=float).reshape(-1,1)])
	# build upper triangle
	#i is the index of the row and column which out pivot should exist on
	for i in range(0, len(mat[0]) - 2):
		#find the value with the largest abs value at a slice starting with our last pivot
		row = np.argmax(np.abs(mat[i:, i])) + i
		#put row at top of the section combed through
		mat[[i, row]] = mat[[row, i]]
		#print("mat after swapping rows:")
		#print(mat)
		# start cancelling stuff out below it 
		#j should be the index of the row that we want to remove things from
		for j in range(i+1, len(mat)):
			if mat[j, i] != 0:
				mat[j, :] = mat[j,:] - (mat[j, i]/mat[i, i]) * mat[i,:]
			#print("mat after doing a subtraction:")
			#print(mat)
		
	#back substitution
	ans = np.zeros(len(mat))
	#i is the row/column index of pivot
	for i in range(len(mat)-1, -1, -1):
		#figure out what the next value has to be
		ans[i] = (mat[i, -1] - sum(mat[i, i+1:-1] * ans[i+1:])) / mat[i, i]


	return ans
