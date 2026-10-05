import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    # get mean and stdev from training split
    mean = np.mean(X_train, axis=0, keepdims=True)
    stdev = np.std(X_train, axis=0, keepdims=True)
    # make sure to replace 0stdev with 1.0
    stdev = np.where(stdev == 0, 1.0, stdev)

    # apply z score normalization to test data
    return (X_test - mean) / stdev
