import numpy as np


def softmax(a):
    # 一番大きい値を取得
    c = np.max(a, axis=0)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=0)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a

    return y