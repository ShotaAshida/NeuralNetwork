import funcs
import numpy as np
from mnist import MNIST

mndata = MNIST("/Users/omushota/ex4-image/le4nn")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)

# 定数 ##############################
# 入力データ関連
line = X.shape[0]
row = X.shape[1]

# バッチ
batch = 100

# 学習
loop = int((len(X) / batch) * 30)
percent = 0.01

# 重み1
middle = 300
average = 0
variance1 = 1.0 / (row * row)
seed = 1
np.random.seed(seed)

weight1 = np.random.normal(average, variance1, (row * row * middle))
weight1 = np.reshape(weight1, (middle, row * row))
b1 = np.random.normal(average, variance1, middle)
b1 = np.reshape(b1, (middle, 1))

# 重み2
end = 10
variance2 = 1.0 / middle

weight2 = np.random.normal(average, variance2, middle * end)
weight2 = np.reshape(weight2, (end, middle))
b2 = np.random.normal(average, variance2, end)
b2 = np.reshape(b2, (end, 1))

# 傾き
aen_ay2 = np.zeros((end, batch))
aen_ax2 = np.zeros((middle, batch))
aen_aw2 = np.zeros((end, middle))
aen_ab2 = np.zeros((end, 1))

aen_ay1 = np.zeros((middle, batch))
aen_ax1 = np.zeros((row*row, batch))
aen_aw1 = np.zeros((middle, row*row))
aen_ab1 = np.zeros((middle, 1))


# 学習 ####################################
for n in range(loop):
    # バッチ選択
    if ((n + 1) * batch) % 60000 != 0:
        learn = np.reshape(X[(n * batch) % 60000: ((n + 1) * batch) % 60000:], (batch, row * row)).T
        answer = Y[(n * batch) % 60000: ((n + 1) * batch) % 60000:]
    else:
        learn = np.reshape(X[(n * batch) % 60000: 60000:], (batch, row * row)).T
        answer = Y[(n * batch) % 60000: 60000:]

    # 中間層
    midin = weight1.dot(learn) + b1
    midout = funcs.sigmoid(midin)

    # 出力層
    finin = weight2.dot(midout) + b2
    finout = funcs.softmax(finin)

    # クロスエントロピー
    entropy = funcs.cross(finout, answer, end)
    if n * batch % 60000 == 0:
        print(str(n) + "回目")
        print(entropy)

    # 逆伝播1
    aen_ay2 = (finout - np.eye(end)[answer].T) / batch
    aen_ax2 = weight2.T.dot(aen_ay2)
    aen_aw2 = aen_ay2.dot(midout.T)
    aen_ab2 = np.reshape(np.sum(aen_ay2, axis=1), (end, 1))

    # 逆伝播2
    aen_ay1 = aen_ax2 * ((1 - midout) * midout)
    aen_ax1 = weight1.T.dot(aen_ay1)
    aen_aw1 = aen_ay1.dot(learn.T)
    aen_ab1 = np.reshape(np.sum(aen_ay1, axis=1), (middle, 1))

    # 重み修正
    weight1 -= percent * aen_aw1
    b1 -= percent * aen_ab1
    weight2 -= percent * aen_aw2
    b2 -= percent * aen_ab2


print("save")
np.savez("parameters.npz", w1=weight1, w2=weight2, b1=b1, b2=b2)
np.savetxt('weight1.csv', weight1, delimiter=',')
np.savetxt('weight2.csv', weight2, delimiter=',')
