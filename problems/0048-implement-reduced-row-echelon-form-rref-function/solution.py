import numpy as np

def rref(matrix):
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    r = 0  # current pivot row
    for c in range(cols):
        if r >= rows:
            break
        # partial pivoting: largest entry at/below row r in this column
        p = r + np.argmax(np.abs(A[r:, c]))
        if abs(A[p, c]) < 1e-10:
            A[r:, c] = 0  # clean up float noise; free column
            continue
        A[[r, p]] = A[[p, r]]   # swap pivot row up
        A[r] /= A[r, c]         # normalize pivot to 1
        for i in range(rows):   # clear above AND below
            if i != r:
                A[i] -= A[i, c] * A[r]
        r += 1
    return A