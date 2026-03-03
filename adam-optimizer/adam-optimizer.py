import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    parameters = np.array(param)
    gradient = np.array(grad)
    mt = np.array(m)
    vt = np.array(v)
    mt = beta1 * mt + (1 - beta1) * gradient
    vt = beta2 * vt + (1 - beta2) * gradient * gradient
    m_hat = mt / (1 - beta1 ** t)
    v_hat = vt / (1 - beta2 ** t)
    parameters = parameters - lr * m_hat / (np.sqrt(v_hat) + eps)
    return (parameters, mt, vt)