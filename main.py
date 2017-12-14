import sys
import sigmoid
import softmax
import math
import numpy as np
from mnist import MNIST



# 入力層
# 画像取り込み
mndata = MNIST("/Users/omushota/Documents/Autumn2017/ex4-image/le4nn")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
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
indata = np.reshape( indata , (row * row, 1) )



# 中間層への入力
# 重み生成
middle = 4
average = 0
variance = math.sqrt(1/line)
np.random.seed(100)

weight = np.random.normal(average, variance, row * row * middle)
weight = np.reshape(weight, (middle, row * row))
b = np.random.normal(average, variance, middle)
b = np.reshape(b, (middle, 1))
midinput = weight.dot(indata) + b



# 中間層
# シグモイド
midout = sigmoid.sigmoid(midinput)



# 出力層への入力
end = 10
average1 = 0
variance1 = math.sqrt(1/middle)

weight1 = np.random.normal(0, variance, middle * end)
weight1 = np.reshape(weight1, (end, middle))
b1 = np.random.normal(average1, variance1, end)
b1 = np.reshape(b1, (end, 1))

fininput = weight1.dot(midout) + b1



# 出力層
# ソフトマックス
finout = softmax.softmax(fininput)
indexmax = np.where(finout == finout.max())[0][0]
print(indexmax)
