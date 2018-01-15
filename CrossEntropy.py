import numpy as np


def cross(a, b, c):
    hotone = np.eye(c)[b].T
    sum1 = np.sum(hotone * (np.log(a) * -1))
    ave = sum1 / b.shape[0]
    return ave
