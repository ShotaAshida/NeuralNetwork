import numpy as np


def cross(a, b):
    hotone = np.eye(10)[b].T
    sum1 = np.sum(hotone * (np.log(a) * -1))
    ave = sum1 / b.shape[0]
    return ave
