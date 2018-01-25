import numpy as np


def cross(a, b, c):
    hotone = np.eye(c)[b].T
    sum1 = np.sum(hotone * (np.log(a) * -1))
    ave = sum1 / b.shape[0]
    return ave


def softmax(a):
    # 一番大きい値を取得
    c = np.max(a, axis=0)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=0)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a
    # print(y)
    return y


@np.vectorize
def sigmoid(x):
    sigmoid_range = 34.538776394910684
    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15
    return 1.0 / (1.0 + np.exp(-x))


@np.vectorize
def ReLU(x):
    if x > 0:
        return x
    else:
        return 0.0

@np.vectorize
def dif_ReLU(x):
    if x > 0:
        return 1
    else:
        return 0
