import pickle
import sigmoid
import softmax
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

# python 3 系の場合は import pickle としてください．
with open("/Users/omushota/ex4-image/le4MNIST_X.dump","rb") as f:
    X = pickle.load(f, encoding='bytes')
    X1 = X.reshape((X.shape[0], 28, 28))
    X = X.reshape((X.shape[0], 28 * 28))
    print(X.shape[1])

weightfile = np.load('parameters2.npz')

line = X.shape[0]
row = X.shape[1]
batch = 10000
loop = int(len(X) / batch)

counter = 0
for n in range(loop):
    print(str(n) + "回目")
    if ((n + 1) * batch) % 10000 != 0:
        learn = np.reshape(X[(n * batch) % 10000: ((n + 1) * batch) % 10000:], (batch, row)).T
        # print(answer)
    else:
        learn = np.reshape(X[(n * batch) % 10000: 10000:], (batch, row)).T

    # 中間層################################
    # 定数
    middle = 300

    # 重み1
    weight1 = weightfile['w1']
    b1 = weightfile['b1']

    # 中間層への入力
    midinput = weight1.dot(learn) + b1

    # シグモイド
    midout = sigmoid.sigmoid(midinput)

    # 出力層##################################
    # 定数
    end = 10

    # 重み2
    weight2 = weightfile['w2']
    b2 = weightfile['b2']

    # 出力層への入力
    fininput = weight2.dot(midout) + b2

    # ソフトマックス
    finout = softmax.softmax(fininput)
    indexmax = finout.argmax(axis=0)
    print("indexmax")
    print(indexmax)

np.savetxt("kaitou_adm.txt", indexmax, fmt="%d")



