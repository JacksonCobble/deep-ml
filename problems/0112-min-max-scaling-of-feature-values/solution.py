def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    x_max = max(x)
    x_min = min(x)
    for idx in range(len(x)):
        x[idx] = (x[idx]-x_min)/(x_max-x_min)

    return x