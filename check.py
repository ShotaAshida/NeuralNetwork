import sigmoid
import softmax
import CrossEntropy
import math
import numpy as np
from mnist import MNIST

a = np.array([[1, 2], [3, 4]])
print(a.max(axis=0))

mndata = MNIST("/Users/omushota/ex4-image/le4nn")
X, Y = mndata.load_testing()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28)) / 255
Y = np.array(Y)

weightfile = np.load('parameters.npz')

line = X.shape[0]
row = X.shape[1]
batch = 100

loop = int(len(X) / batch)

counter = 0
for n in range(loop):
    if ((n + 1) * batch) % 10000 != 0:
        learn = np.reshape(X[(n * batch) % 10000: ((n + 1) * batch) % 10000:], (row * row, batch))
        answer = Y[(n * batch) % 10000: ((n + 1) * batch) % 10000:]
    else:
        learn = np.reshape(X[(n * batch) % 10000: 10000:], (row * row, batch))
        answer = Y[(n * batch) % 10000: 10000:]

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
    power = indexmax - answer
    counter = counter + len(np.where(power == 0)[0])
    print(counter)

print((counter / 10000) * 100)

