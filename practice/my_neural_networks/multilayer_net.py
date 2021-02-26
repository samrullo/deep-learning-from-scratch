import numpy as np
from practice.layers import Affine, Relu, Sigmoid, SoftmaxWithLoss


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, activation, weight_init_std, weight_decay_lambda):
        """
        MultiLayer neural network initialization
        :param input_size: for the case of MNIST it is 28*28=784
        :param hidden_size_list: list of the number of neurons in each hidden layer
        :param output_size: output size matches number of classes, in case of MNIST it is 10
        :param activation: activation function name 'relu' or 'sigmoid'
        :param weight_init_std: it can be either a number or a string with values like 'he','xavier'
        :param weight_decay_lambda:
        """
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.activation = activation
        self.weight_decay_lambda = weight_decay_lambda

        self.params = {}

        # initialize weights
        self.__init_weights(weight_init_std)

        # activation layers dictionary
        activation_layers = {'sigmoid': Sigmoid, 'relu': Relu}

        # set layers
        layers = {}
        for idx in range(1, len(hidden_size_list) + 1):
            layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            layers['Activation' + str(idx)] = activation_layers[self.activation]()
        layers['Affine' + str(idx + 1)] = Affine(self.params['W' + str(idx + 1)], self.params['b' + str(idx + 1)])
        self.layers = layers
        self.lastLayer = SoftmaxWithLoss()

    def __init_weights(self, weight_init_std):
        """
        Initialize weights. There is gonna be two dimensional W and one dimensional bias parameters per hidden layer.
        As each hidden activation layer will have Affine layer connected to it
        :param weight_init_std: either a number such as 0.01 or string like 'xavier' or 'he'
        :return:
        """

        # first make a list that contains number of neurons in each layer including input_size and output_size
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        scale = weight_init_std

        # there are going to be total of len(hidden_size_list)+1 Affine layers
        # since range(len(hidden_size_list)) last index is gonna be len(hidden_size_list)-1
        # and we want to start with W1, we loop with range(1, len(hidden_size_list)+2)
        for idx in range(1, len(self.hidden_size_list) + 2):
            if str(scale) in ['sigmoid', 'xavier']:
                scale = np.sqrt(1 / all_size_list[idx - 1])
            elif str(scale) in ['relu', 'he']:
                scale = np.sqrt(2 / all_size_list[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        """
        Predict the label of the input x
        :param x:
        :return:
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        y_labels = np.argmax(y, axis=1)
        t_labels = np.argmax(t, axis=1)
        return np.mean(y_labels == t_labels)

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # forward once
        self.loss(x, t)

        layers = list(self.layers.values())
        layers.reverse()

        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)

        # dictionary that holds gradients of all parameters
        grads = {}

        for idx in range(1, len(self.hidden_size_list) + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
