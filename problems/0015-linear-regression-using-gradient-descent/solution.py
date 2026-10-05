import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1) # ensure y is a column vector
    theta = np.zeros((n, 1)) # Initialize weights to zeros

    # Your code here: implement gradient descent
    # loop over iterations
    for i in range(iterations):
        # compute predictions
        yhat = X @ theta #mxn * nx1 = mx1 column vector
        # compute errors
        e = yhat - y #e=mx1 column vector
        # compute gradient
        g = 1/m * (X.T @ e) #nxm * mx1 = nx1 column vector
        # update weights
        theta = theta - (alpha * g)


    return theta.flatten()