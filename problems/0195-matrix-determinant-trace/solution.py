def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.

    Args:
        matrix: A square matrix (n x n) represented as list of lists

    Returns:
        Tuple of (determinant, trace)
    """
    n = len(matrix)

    # Trace: sum of the main diagonal
    trace = sum(matrix[i][i] for i in range(n))

    def det(m):
        size = len(m)

        # Base cases
        if size == 1:
            return m[0][0]
        if size == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        # Cofactor expansion along row 0
        total = 0
        for j in range(size):
            # Minor: drop row 0 and column j
            minor = [row[:j] + row[j+1:] for row in m[1:]]
            # Sign alternates (+, -, +, ...) with column index
            total += ((-1) ** j) * m[0][j] * det(minor)
        return total

    return det(matrix), trace