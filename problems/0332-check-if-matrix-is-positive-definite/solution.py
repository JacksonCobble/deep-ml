import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    #build structure program wants us to use
    mydict = {
        "is_positive_definite": False,
        "eigenvalues": None
    }

    #calculate eigenvalues
    matrix = np.array(matrix, dtype=float)
    evals = np.linalg.eigvals(matrix).round(4)
    evals.sort()
    mydict["eigenvalues"] = evals.tolist()

    #matrix pos definite if all eigenvalues > 0
    if np.all(evals > 0):
        mydict["is_positive_definite"] = True

    return mydict