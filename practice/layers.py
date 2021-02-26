import numpy as np
from practice.activation import softmax
from practice.loss_functions import cross_entrpy_error


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = self.x.dot(self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = self.x.T.dot(dout)
        self.db = dout.sum(axis=0)
        dx = dout.dot(self.W.T)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.t = None
        self.y = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entrpy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


