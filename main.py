import sigmoid
import softmax
import  CrossEntropy
import math
import numpy as np
from mnist import MNIST
from layer import mid, endend


# 入力層
# 画像取り込み
mndata = MNIST("/Users/omushota/ex4-image/le4nn")
X, Y = mndata.load_testing()
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
        # print("break")
        break

indata = X[num]
line = X.shape[0]
row = X.shape[1]
indata = np.reshape(indata, (row * row, 1))


# 中間層
middle = 4
average = 0
variance = math.sqrt(1/line)
seed = 100
midinput = mid(indata, middle, row, average, variance, seed)

# シグモイド
midout = sigmoid.sigmoid(midinput)


# 出力層
end = 10
average1 = 0
variance1 = math.sqrt(1/middle)
fininput = endend(midout, end, middle, average1, variance1, seed)

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
minimidinput = mid(minidata, middle, row, average, variance, seed)
minimidout = sigmoid.sigmoid(minimidinput)
minifininput = endend(minimidout, end, middle, average1, variance1, seed)
minifinout = softmax.softmax(minifininput)


# クロスエントロピー
minilogs = np.log(minifinout) * -1
entropy = CrossEntropy.cross(minilogs, minianswer)
crossave = np.average(entropy)
print(crossave)

