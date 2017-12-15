import sigmoid
import softmax
import CrossEntropy
import math
import numpy as np
from mnist import MNIST


# 入力層#############################
# 画像取り込み
mndata = MNIST("/Users/omushota/ex4-image/le4nn")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)

# キーボード入力待ち
while True:
    strnum = input("input number : ")
    num = int(strnum)
    if (num < 0) or (num > 9999):
        print("Please type 0 ~ 9999")
    else:
        break

indata = X[num]
line = X.shape[0]
row = X.shape[1]
indata = np.reshape(indata, (row * row, 1))


# 中間層################################
# 定数
middle = 4
average1 = 0
variance1 = math.sqrt(1 / line)
seed = 100

# 重み1
np.random.seed(seed)
weight1 = np.random.normal(average1, variance1, row * row * middle)
weight1 = np.reshape(weight1, (middle, row * row))
b1 = np.random.normal(average1, variance1, middle)
b1 = np.reshape(b1, (middle, 1))

# 中間層への入力
midinput = weight1.dot(indata) + b1

# シグモイド
midout = sigmoid.sigmoid(midinput)


# 出力層##################################
# 定数
end = 10
average2 = 0
variance2 = math.sqrt(1/middle)

# 重み2
np.random.seed(seed)
weight2 = np.random.normal(average2, variance2, middle * end)
weight2 = np.reshape(weight2, (end, middle))
b2 = np.random.normal(average2, variance2, end)
b2 = np.reshape(b2, (end, 1))
fininput = weight2.dot(midout) + b2

# ソフトマックス
finout = softmax.softmax(fininput)
indexmax = np.where(finout == finout.max())[0][0]
print(indexmax)


# 課題2
# ミニバッチ
batch = 100
choice = np.random.choice(len(X), batch, replace=False)
minidata = np.reshape(X[choice], (batch, row*row))
minidata = minidata.T
minianswer = Y[choice]

# ミニバッチにニューラル適用
minimidinput = weight1.dot(minidata)
minimidout = sigmoid.sigmoid(minimidinput)
minifininput = weight2.dot(minimidout)
minifinout = softmax.softmax(minifininput)

# クロスエントロピー
minilogs = np.log(minifinout) * -1
entropy = CrossEntropy.cross(minilogs, minianswer)
crossave = np.average(entropy)
print(crossave)


# 課題3
# ソフトマックス + 損失関数 の逆伝播　w2, b2の導出
aen_ay = (minifinout - (np.eye(10)[minianswer]).T) / batch
aen_ax = weight2.T.dot(aen_ay)
aen_aw2 = aen_ay.dot(minimidout.T)
aen_ab2 = np.sum(aen_ay, axis=1)

# シグモイドの逆伝播 W1, b1の導出
aen_ay1 = sigmoid.difsigmoid(aen_ax)
ae_ax1 = weight1.T.dot(aen_ay1)
aen_aw1 = aen_ay1.dot(minidata.T)
aen_ab1 = np.sum(aen_ay1, axis=1)

