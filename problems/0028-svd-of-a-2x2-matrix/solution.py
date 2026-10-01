import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    a, b = A[0,0] , A[0,1]
    c, d = A[1,0] , A[1,1]

    # compute auxillary values
    y = np.array([c + b, c - b], dtype=float)
    x = np.array([a - d, a + d], dtype=float)

    # compute norms of each part
    h = np.hypot(x, y)

    # compute singular values
    sigma = np.array([(h[0] + h[1]) / 2, np.abs(h[0] - h[1]) / 2])

    # angles for computing U/Vt
    gamma = np.arctan2(y[0], x[0])
    beta = np.arctan2(y[1], x[1])
    p = (beta + gamma) / 2
    q = (beta - gamma) / 2

    U = np.array([[np.cos(p), -np.sin(p)],
                [np.sin(p), np.cos(p)]])
    Vt = np.array([[np.cos(q), -np.sin(q)],
                [np.sin(q), np.cos(q)]])

    # if h1>h2, the raw second diagonal entry is negative: flip a column of U
    
    if h[0] > h[1]:
        U[:,1] *= -1

    return U, sigma, Vt