import numpy as np
from practice.activation import sigmoid, softmax
from practice.numerical_gradient import numerical_gradient
from practice.loss_functions import cross_entrpy_error
import datetime
import matplotlib.pyplot as plt
from practice.layers import Affine, Relu, SoftmaxWithLoss


class TwoLayerNetBackprop:
    def __init__(self, weight_init_std, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = {}
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y_labels = y.argmax(axis=1)
        t_labels = t.argmax(axis=1)
        return np.mean(y_labels == t_labels)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grad = {}
        grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grad

    def gradient(self, x, t):
        # forward all the layers once
        self.loss(x, t)

        layers = list(self.layers.values())
        layers.reverse()

        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
