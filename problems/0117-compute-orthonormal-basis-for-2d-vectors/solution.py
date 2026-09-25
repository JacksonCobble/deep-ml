import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    vectors = np.array(vectors, dtype=float)

    #list to make up orthonormal basis
    u: list[np.ndarray] = []

    #go over each vector to try to make orthagonal
    for idx in range(len(vectors)):

        sum_of_proj = np.zeros(len(vectors[0]))
        #get summation of vector projections proj u V
        for u_vec in u:
            sum_of_proj += np.dot(vectors[idx], u_vec) * u_vec

        #subtract projections to get w
        w = vectors[idx] - sum_of_proj

        #normalize if length of vector > some min value (just checking to see if its not a 0 vector or close to it, because that would mean its a linearly dependent vector) and then if so, append it to the list of vectors in our orthonormal basis
        if np.linalg.norm(w) > tol:
            u.append(w / np.linalg.norm(w))

    return u