import numpy as np
import pickle
# python 3 系の場合は import pickle としてください．
with open("/Users/omushota/ex4-image/le4MNIST_X.dump","rb") as f:
    X = pickle.load(f, encoding='bytes')
    X = X.reshape((X.shape[0], 28, 28))
    X = X.reshape((X.shape[0], 28 * 28))
    print(X.shape)

