import funcs
import numpy as np
from mnist import MNIST

mndata = MNIST("/Users/omushota/ex4-image/le4nn")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)

# 定数 ##############################
# データ関連
line = X.shape[0]
row = X.shape[1]
# row = 2

# バッチ
batch = 100
# batch = 5

# 学習
loop = int((len(X) / batch) * 25)
# loop = 10

# RMSPROP
# percent = 0.001
# p = 0.90
# e = 0.00000001
# hw1 = 0
# hw2 = 0
# hb1 = 0
# hb2 = 0

# 重み1
middle = 300
# middle = 1
average = 0
variance1 = 1.0 / (row * row)
# variance1 = 1 / 784
# print(variance1)

seed = 1
np.random.seed(seed)
weight1 = np.random.normal(average, variance1, (row * row * middle))
weight1 = np.reshape(weight1, (middle, row * row))
b1 = np.random.normal(average, variance1, middle)
b1 = np.reshape(b1, (middle, 1))

# print("weight1")
# print(weight1)
# print("b1")
# print(b1)

# 重み2
end = 10
# end = 2
variance2 = 1.0 / middle
# variance2 = 1 / 300
# print(variance2)

weight2 = np.random.normal(average, variance2, middle * end)
weight2 = np.reshape(weight2, (end, middle))
b2 = np.random.normal(average, variance2, end)
b2 = np.reshape(b2, (end, 1))

# print("weight2")
# print(weight2)
# print("b2")
# print(b2)
# print(" ")
# print(" ")

# 傾き
aen_ay2 = np.zeros((end, batch))
aen_ax2 = np.zeros((middle, batch))
aen_aw2 = np.zeros((end, middle))
aen_ab2 = np.zeros((end, 1))

aen_ay1 = np.zeros((middle, batch))
aen_ax1 = np.zeros((row*row, batch))
aen_aw1 = np.zeros((middle, row*row))
aen_ab1 = np.zeros((middle, 1))

# 学習方法
# # adam
m_w1 = np.zeros((middle, row*row))
v_w1 = np.zeros((middle, row*row))

m_b1 = np.zeros((middle, 1))
v_b1 = np.zeros((middle, 1))

m_w2 = np.zeros((end, middle))
v_w2 = np.zeros((end, middle))

m_b2 = np.zeros((end, 1))
v_b2 = np.zeros((end, 1))

alpha = 0.001
beta1 = 0.9
beta2 = 0.999
e = 0.00000001


# 学習 ####################################
for n in range(loop):
    # print(str(n) + "回目")
    # print((n * batch) % 60000)
    # print(((n + 1) * batch) % 60000)

    # バッチ選択
    # learn = np.array([[0], [16], [255], [121]])
    # answer = np.array([1])
    if ((n + 1) * batch) % 60000 != 0:
        learn = np.reshape(X[(n * batch) % 60000: ((n + 1) * batch) % 60000:], (batch, row * row)).T
        answer = Y[(n * batch) % 60000: ((n + 1) * batch) % 60000:]
        # print(answer)
    else:
        learn = np.reshape(X[(n * batch) % 60000: 60000:], (batch, row * row)).T
        answer = Y[(n * batch) % 60000: 60000:]
        # print(answer)
    # 中間層
    midin = weight1.dot(learn) + b1
    midout = funcs.sigmoid(midin)

    # 出力層
    finin = weight2.dot(midout) + b2
    finout = funcs.softmax(finin)

    indexmax = finout.argmax(axis=0)
    power = indexmax - answer

    # print("answer")
    # print(answer)
    # print("indexmax")
    # print(indexmax)
    # print("power")
    # print(power)

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

    # print("midin")
    # print(midin)
    # print("midout")
    # print(midout)
    # print("finin")
    # print(finin)
    # print("finout")
    # print(finout)
    # print("1-midmid")
    # print(((1 - midout) * midout))
    # print(" ")
    # print("learn")
    # print(learn)

    # print("aen_ay1")
    # print(aen_ay1)
    # print("aen_ay2")
    # print(aen_ay2)
    # print("aen_ax1")
    # print(aen_ax1)
    # print("aen_ax2")
    # print(aen_ax2)
    #
    # print("aen_aw1")
    # print(aen_aw1)
    # print("aen_aw2")
    # print(aen_aw2)
    # print("ave aw2")
    # print(np.average(aen_aw2 * aen_aw2))
    # print(" ")
    # print(" ")

    # print("weight1")
    # print(weight1)
    # print("b1")
    # print(b1)
    # print("weight2")
    # print(weight2)
    # print("b2")
    # print(b2)

    # 重み修正
    # RMSprop
    # weight1 -= (percent / (math.sqrt(hw1 + e))) * aen_aw1
    # b1 -= (percent / (math.sqrt(hb1 + e))) * aen_ab1
    # weight2 -= (percent / (math.sqrt(hw2 + e))) * aen_aw2
    # b2 -= (percent / (math.sqrt(hb2 + e))) * aen_ab2
    #
    # hw1 = p * hw1 + (1 - p) * np.average(aen_aw1 * aen_aw1)
    # hb1 = p * hb1 + (1 - p) * np.average(aen_ab1 * aen_ab1)
    # hw2 = p * hw2 + (1 - p) * np.average(aen_aw2 * aen_aw2)
    # hb2 = p * hb2 + (1 - p) * np.average(aen_ab2 * aen_ab2)

    # adam
    m_w1 = beta1 * m_w1 + (1 - beta1) * aen_aw1
    v_w1 = beta2 * v_w1 + (1 - beta2) * aen_aw1 * aen_aw1
    m_w1_dash = m_w1 / (1 - beta1 ** (n + 1))
    v_w1_dash = v_w1 / (1 - beta2 ** (n + 1))
    weight1 -= alpha * m_w1_dash / (np.sqrt(v_w1_dash) + e)

    m_b1 = beta1 * m_b1 + (1 - beta1) * aen_ab1
    v_b1 = beta2 * v_b1 + (1 - beta2) * aen_ab1 * aen_ab1
    m_b1_dash = m_b1 / (1 - beta1 ** (n + 1))
    v_b1_dash = v_b1 / (1 - beta2 ** (n + 1))
    b1 -= alpha * m_b1_dash / (np.sqrt(v_b1_dash) + e)

    m_w2 = beta1 * m_w2 + (1 - beta1) * aen_aw2
    v_w2 = beta2 * v_w2 + (1 - beta2) * aen_aw2 * aen_aw2
    m_w2_dash = m_w2 / (1 - beta1 ** (n + 1))
    v_w2_dash = v_w2 / (1 - beta2 ** (n + 1))
    weight2 -= alpha * m_w2_dash / (np.sqrt(v_w2_dash) + e)

    m_b2 = beta1 * m_b2 + (1 - beta1) * aen_ab2
    v_b2 = beta2 * v_b2 + (1 - beta2) * aen_ab2 * aen_ab2
    m_b2_dash = m_b2 / (1 - beta1 ** (n + 1))
    v_b2_dash = v_b2 / (1 - beta2 ** (n + 1))
    b2 -= alpha * m_b2_dash / (np.sqrt(v_b2_dash) + e)

    # print(entropy)
    # print(" ")
    # print(" ")

print("save")
np.savez("parameters2.npz", w1=weight1, w2=weight2, b1=b1, b2=b2)
np.savetxt('weight1-2.csv', weight1, delimiter=',')
np.savetxt('weight2-2.csv', weight2, delimiter=',')
