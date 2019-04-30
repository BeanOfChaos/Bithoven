import numpy as np


def sigmoid(value, derivative=False):
    """
    Basic Sigmoid calculation
    """
    # return np.exp(value)/(np.exp(value) + 1)
    if derivative:
        sigm = sigmoid(value)
        return sigm * (1 - sigm)
    else:
        return 1/(np.exp(-value) + 1)


def reLu(x, derivative=False):
    """Rectified linear unit (ReLu) function
    """
    if derivative:
        return x > 0  # True == 1 and False == 0
    else:
        return x * (x > 0)
