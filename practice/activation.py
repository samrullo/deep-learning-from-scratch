import numpy as np


def softmax(a):
    a = a - a.max(axis=-1, keepdims=True)
    exp_a = np.exp(a)
    exp_sum_a = np.sum(exp_a, axis=-1, keepdims=True)
    return exp_a / exp_sum_a


def sigmoid(a):
    a = a - a.max(axis=-1, keepdims=True)
    return 1 / (1 + np.exp(-a))


if __name__ == '__main__':
    a = np.random.randn(100, 3)
    y = softmax(a)
    print(y[0].sum())
    y_sigmoid = sigmoid(a)
    print(y_sigmoid[0])
