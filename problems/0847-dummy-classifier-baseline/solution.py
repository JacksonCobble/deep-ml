import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    histo = Counter(y_train)
    unique = np.unique(y_train)
    unique.sort()

    match strategy:
        case "most_frequent":
            top_item, _ = histo.most_common(1)[0]
            return [top_item for _ in range(n_test)]
        case "constant":
            return [constant for _ in range(n_test)]
        case "uniform":
            return [unique[i % len(unique)] for i in range(n_test)]
        case "stratified":
            # exact integer math: quota_c = n_test * count_c / n_train
            floors = {c: (n_test * histo[c]) // len(y_train) for c in unique}
            remainders = {c: (n_test * histo[c]) % len(y_train) for c in unique}

            leftover = n_test - sum(floors.values())
            for c in sorted(unique, key=lambda c: (-remainders[c], c))[:leftover]:
                floors[c] += 1

            return [c for c in unique for _ in range(floors[c])]

