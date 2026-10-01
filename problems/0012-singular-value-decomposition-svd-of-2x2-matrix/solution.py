import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    #compute AtA
    b = np.array([[np.linalg.norm(A[:, 0])**2, np.dot(A[:, 0], A[:, 1])],
                    [np.dot(A[:,0], A[:,1]), np.linalg.norm(A[:,1])**2]], dtype=float)
    
    theta = 1/2 * np.arctan2(2 * b[0][1], b[0][0] - b[1][1])
    
    #build rotation matrix
    r = np.array([[np.cos(theta), -1*np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], dtype=float)
    
    #get diagonal matrix
    d = r.T @ b @ r
    eigenvalues = np.diag(d)
    sing_values = np.sqrt(eigenvalues)
    # compute U
    sigma = np.diag(np.array([1/sing_values[0], 1/sing_values[1]], dtype=float))
    # v = r
    u = A @ r @ sigma
    
    return u, sing_values, r.T
