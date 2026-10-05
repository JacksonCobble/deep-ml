import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    #get number of data points
    n = data.shape[0]
    # get randomindex order of datapoints based on rng seed
    idx = np.random.default_rng(seed).permutation(n)
    # find indicies of split ends based on fracs
    train_end = int(n * train_frac)
    validation_end = train_end + int(n*validation_frac)

    # get indexes of each of the splits
    train_idx = idx[:train_end]
    val_idx = idx[train_end:validation_end]
    test_idx = idx[validation_end:]

    #return our splits with fancy np indexing
    return [data[train_idx], data[val_idx], data[test_idx]]
