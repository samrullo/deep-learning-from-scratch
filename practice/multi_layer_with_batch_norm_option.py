from dataset.mnist import load_mnist
from practice.layers import Affine, Relu, SoftmaxWithLoss
from practice.batch_normalization import BatchNormalization
import numpy as np
import re
from collections import OrderedDict


class MultiLayerNet:
    def __init__(self, weight_init_std, input_size, node_size, output_size, layer_size, use_batch_norm=True):
        self.layer_size = layer_size
        self.use_batch_norm = use_batch_norm
        self.weight_init_std = weight_init_std
        self.input_size = input_size
        self.node_size = node_size
        self.output_size = output_size
        self.params = {}
        self.layers = OrderedDict()
        if use_batch_norm:
            self.__init_params_and_layers_with_batch_norm()
        else:
            self.__init_params_and_layers_without_batch_norm()

    def __init_params_and_layers_without_batch_norm(self):

        self.params["W1"] = self.weight_init_std * np.random.randn(self.input_size, self.node_size)
        self.params["b1"] = np.zeros(self.node_size)
        self.layers["affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["relu1"] = Relu()
        for i in range(1, self.layer_size):
            self.params[f'W{i + 1}'] = self.weight_init_std * np.random.randn(self.node_size, self.node_size)
            self.params[f'b{i + 1}'] = np.zeros(self.node_size)
            self.layers[f'affine{i + 1}'] = Affine(self.params[f'W{i + 1}'], self.params[f'b{i + 1}'])
            self.layers[f'relu{i + 1}'] = Relu()
        self.params[f"W{self.layer_size + 1}"] = self.weight_init_std * np.random.randn(self.node_size,
                                                                                        self.output_size)
        self.params[f"b{self.layer_size + 1}"] = np.zeros(self.output_size)
        self.layers[f"affine{self.layer_size + 1}"] = Affine(self.params[f"W{self.layer_size + 1}"],
                                                             self.params[f"b{self.layer_size + 1}"])
        self.lastLayer = SoftmaxWithLoss()

    def __init_params_and_layers_with_batch_norm(self):
        self.params["W1"] = self.weight_init_std * np.random.randn(self.input_size, self.node_size)
        self.params["b1"] = np.zeros(self.node_size)
        self.layers["affine1"] = Affine(self.params["W1"], self.params["b1"])

        # initialize gamma and beta params
        self.params["gamma1"] = np.ones(self.node_size)
        self.params["beta1"] = np.zeros(self.node_size)
        self.layers['batch_norm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'], 0.9)
        self.layers["relu1"] = Relu()
        for i in range(1, self.layer_size):
            self.params[f'W{i + 1}'] = self.weight_init_std * np.random.randn(self.node_size, self.node_size)
            self.params[f'b{i + 1}'] = np.zeros(self.node_size)
            self.layers[f'affine{i + 1}'] = Affine(self.params[f'W{i + 1}'], self.params[f'b{i + 1}'])

            # initialize gamma and beta params
            self.params[f"gamma{i + 1}"] = np.ones(self.node_size)
            self.params[f"beta{i + 1}"] = np.zeros(self.node_size)
            self.layers[f'batch_norm{i + 1}'] = BatchNormalization(self.params[f'gamma{i + 1}'],
                                                                   self.params[f'beta{i + 1}'], 0.9)
            self.layers[f'relu{i + 1}'] = Relu()
        self.params[f"W{self.layer_size + 1}"] = self.weight_init_std * np.random.randn(self.node_size,
                                                                                        self.output_size)
        self.params[f"b{self.layer_size + 1}"] = np.zeros(self.output_size)
        self.layers[f"affine{self.layer_size + 1}"] = Affine(self.params[f"W{self.layer_size + 1}"],
                                                             self.params[f"b{self.layer_size + 1}"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, training_flg=True):
        for key, layer in self.layers.items():
            if re.search('batch_norm', key):
                x = layer.forward(x, training_flg=training_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, training_flg=False)
        y_labels = np.argmax(y, axis=1)
        t_labels = np.argmax(t, axis=1)
        accuracy = np.sum(y_labels == t_labels) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        layers = list(self.layers.values())
        layers.reverse()

        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i in range(self.layer_size + 1):
            grads[f"W{i + 1}"] = self.layers[f"affine{i + 1}"].dW
            grads[f"b{i + 1}"] = self.layers[f"affine{i + 1}"].db
        if self.use_batch_norm:
            for i in range(self.layer_size):
                grads[f"gamma{i + 1}"] = self.layers[f"batch_norm{i + 1}"].dgamma
                grads[f"beta{i + 1}"] = self.layers[f"batch_norm{i + 1}"].dbeta
        return grads
