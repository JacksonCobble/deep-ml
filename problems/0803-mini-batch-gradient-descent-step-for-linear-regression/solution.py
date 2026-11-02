import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    batch_size = len(batch_indices)
    w = np.array(weights, dtype=float)
    b = bias

    # slice training data and answers
    x_slice = X[batch_indices, :]
    y_slice = y[batch_indices]
    
    # compute predictions
    pred = (x_slice @ w) + bias

    # compute error vector
    e = pred-y_slice

    # compute gradient
    w_grad = (2/batch_size) * (x_slice.T @ e)
    b_grad = (2/batch_size) * np.sum(e)

    # update weights and bias
    w -= lr * w_grad
    b -= lr * b_grad

    return np.append(w, b)
    
