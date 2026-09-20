import numpy as np

def matrix_image(A):
    A = np.array(A, dtype=float)
    M = A.copy()
    rows, cols = M.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # pick the largest entry in this column at/below row r
        p = r + np.argmax(np.abs(M[r:, c]))
        if abs(M[p, c]) < 1e-10:
            continue  # no pivot in this column -> free column
        M[[r, p]] = M[[p, r]]  # row swap
        for i in range(r + 1, rows):  # clear BELOW only
            M[i] -= (M[i, c] / M[r, c]) * M[r]
        pivot_cols.append(c)
        r += 1
    return A[:, pivot_cols]  # from the ORIGINAL A
