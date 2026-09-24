import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	#initialize vectors
	A = np.array(A, dtype=float)
	b = np.array(b, dtype=float)

	# set x to initial guess
	x = None
	if x0 == None:
		x = np.zeros(len(b))
	else:
		x = x0
	
	#calculate initial residual
	r = b - A @ x

	#initial search direction
	p = r

	it = 0
	#iterate
	while it < n: 
		#compute step size
		alpha = (r.T @ r)/(p.T @ A @ p)
		#update solution
		x = x + (alpha*p)
		#update residual
		r_new = r - (alpha * A @ p)

		#check to see if weve converged
		if np.linalg.norm(r_new) < tol:
			break
		
		#calc new direction scaling
		beta = (r_new.T @ r_new)/ (r.T @ r)

		# update search direction
		p = r_new + (beta * p)

		#update residual to reflect new value
		r = r_new

		#increase iteration
		it+=1

	return x
