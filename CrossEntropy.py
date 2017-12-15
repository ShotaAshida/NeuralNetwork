import numpy as np


def cross(a, b):
    hotone = np.eye(10)[b]
    return np.diag(hotone.dot(a))
