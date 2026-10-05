import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """

    match strategy:
        #return arethmetic mean of ytrain 
        case "mean":
            mean = np.mean(y_train)
            return [mean for _ in range(n_test)]

        #return arethmetic median of ytrain
        case "median":
            median = np.median(y_train)
            return [median for _ in range(n_test)]

        #n-th quantile of ytrain using lerp
        case "quantile":
            if quantile is None or quantile < 0 or quantile > 1:
                raise ValueError("Quantile is set to None or not in range [0,1]")
            quant = np.quantile(y_train, quantile, method='linear')
            return [quant for _ in range(n_test)]
            
        #return constant value in params
        case "constant":
            if constant is None:
                raise ValueError("Constant is set to None")
            return [constant for _ in range(n_test)]

        #base case raise valueerror
        case _:
            raise ValueError("unknown strategy")
    pass
