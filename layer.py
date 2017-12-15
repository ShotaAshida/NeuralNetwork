import numpy as np


def mid(indata, middle, row, average, variance, seed):
    np.random.seed(seed)
    weight = np.random.normal(average, variance, row * row * middle)
    weight = np.reshape(weight, (middle, row * row))
    b = np.random.normal(average, variance, middle)
    b = np.reshape(b, (middle, 1))
    return weight.dot(indata) + b


def endend(midout, end, middle, average, variance, seed):
    np.random.seed(seed)
    weight1 = np.random.normal(0, variance, middle * end)
    weight1 = np.reshape(weight1, (end, middle))
    b1 = np.random.normal(average, variance, end)
    b1 = np.reshape(b1, (end, 1))
    return weight1.dot(midout) + b1
