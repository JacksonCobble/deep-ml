import numpy as np

def lu_decomposition(A):
    """
    Factor square matrix A into L (lower, unit diagonal) and U (upper)
    so that A = L @ U.

    Core idea: A = LU is a system of n^2 equations (one per entry of A).
    If we visit entries in the right order, each equation has exactly ONE
    unknown, because everything else was computed on an earlier iteration.
    """
    A = np.array(A, dtype=float)
    n = len(A)

    # L starts as the identity. This gives us the unit diagonal for free
    # (Doolittle convention) and pre-fills the zeros above the diagonal.
    # Only the strictly-lower part gets overwritten below.
    L = np.eye(n)

    # U starts as zeros. The zeros below the diagonal are never touched,
    # because we only ever write to U[i, j] with j >= i.
    U = np.zeros((n, n))

    # Outer loop: each iteration i finalizes ROW i of U and COLUMN i of L.
    # After iteration i, the top-left (i+1)x(i+1) block is fully done,
    # and everything we need for later iterations is already available.
    for i in range(n):

        # ---------- Part 1: compute row i of U ----------
        # Derivation: A[i][j] = sum over k of L[i][k] * U[k][j].
        # L is lower triangular, so L[i][k] = 0 for k > i, and L[i][i] = 1.
        # Split the sum at k = i:
        #   A[i][j] = (sum for k < i of L[i][k]*U[k][j]) + 1*U[i][j]
        # Rearranged:
        #   U[i][j] = A[i][j] - (sum for k < i of L[i][k]*U[k][j])
        # Every term on the right uses L entries from columns < i and
        # U entries from rows < i, all finished in earlier iterations.
        # Only j >= i is computed, since U is zero below the diagonal.
        for j in range(i, n):
            # L[i, :i] = the already-known multipliers in row i of L
            # U[:i, j] = the already-known entries above U[i][j] in column j
            # The @ (dot product) computes the summation in one shot.
            # When i = 0 both slices are empty, so the sum is 0 and
            # U[0, j] = A[0, j]  (the first row of U is just the first row of A).
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]

        # ---------- Part 2: compute column i of L ----------
        # Same equation, but now solving for L[j][i] where j > i (below diagonal).
        # A[j][i] = sum over k of L[j][k] * U[k][i].
        # U is upper triangular, so U[k][i] = 0 for k > i. Split at k = i:
        #   A[j][i] = (sum for k < i of L[j][k]*U[k][i]) + L[j][i]*U[i][i]
        # Rearranged:
        #   L[j][i] = (A[j][i] - sum for k < i of L[j][k]*U[k][i]) / U[i][i]
        # U[i][i] is the PIVOT, and we just computed it in Part 1.
        # That's why Part 1 must come before Part 2 in each iteration.
        for j in range(i + 1, n):
            # Numerator: A entry minus the contributions of earlier columns.
            # L[j, :i] = row j of L, columns before i (already known)
            # U[:i, i] = column i of U, rows before i (already known)
            # Dividing by the pivot is exactly the "multiplier" from
            # Gaussian elimination: (entry to eliminate) / (pivot).
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
            #                                             ^^^^^^^^
            # DANGER: if this is 0 (or tiny), the algorithm blows up.
            # That's the reason pivoting (PA = LU) exists.

    return L, U