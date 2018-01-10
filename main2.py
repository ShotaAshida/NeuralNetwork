import sigmoid
import softmax
import CrossEntropy
import math
import numpy as np
from mnist import MNIST



mndata = MNIST("/Users/omushota/ex4-image/le4nn")
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0], 28, 28))
Y = np.array(Y)