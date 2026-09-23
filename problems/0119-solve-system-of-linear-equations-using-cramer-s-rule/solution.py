import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    ans = np.zeros(len(A[0]))
    det_a = np.linalg.det(A)
    if det_a == 0:
        return -1
        
    for col in range(len(ans)):
        new_a = A.copy()
        new_a[:, col] = b 

        ans[col] = np.linalg.det(new_a)/det_a
    
    return ans