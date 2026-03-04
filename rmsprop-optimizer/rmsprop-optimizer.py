import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    W = np.array(w)
    G = np.array(g)
    S = np.array(s)
    S = beta * S + (1 - beta) * G * G
    W = W - lr / np.sqrt(S + eps) * G
    return (W, S)