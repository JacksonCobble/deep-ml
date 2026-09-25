import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.

    #normal equation approach 
    #make col of 1s to fold in the bias feature
    ones = np.ones((X.shape[0], 1))
    #put cols side by side, make first weight the bias weight
    X_bias = np.concatenate([ones, X], axis=1)
    #solve using lstsq- uses SVD based pseudoinverse, computes moore-penrose in a way that handles rank deficient and illconditioned X for us. o index because it returns 4 different things. we only care about solution(index 0)
    theta = np.linalg.lstsq(X_bias, y, rcond=None)[0]
    b, W = theta[0], theta[1:]
    return W, b
