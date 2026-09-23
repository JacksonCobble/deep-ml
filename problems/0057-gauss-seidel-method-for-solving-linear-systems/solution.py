import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	mat = np.hstack((np.array(A, dtype=float), np.array(b, dtype=float).reshape(-1,1)))
	ans = np.zeros(len(mat[0]) - 1)

	# iteration
	for i in range(n):
		#represents pivot row/col
		for j in range(len(mat)):
			left_sum = np.dot(ans[0:j], mat[j, 0:j])
			right_sum = np.dot(ans[j+1:], mat[j,j+1:-1])

			ans[j] = (mat[j, -1] - left_sum - right_sum) / mat[j,j]

	return ans
