import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    X = np.array(x)
    return 1.0 / (1 + np.exp(-X))