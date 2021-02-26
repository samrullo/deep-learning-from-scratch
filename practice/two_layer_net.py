import numpy as np
from practice.activation import sigmoid, softmax
from practice.numerical_gradient import numerical_gradient
from practice.loss_functions import cross_entrpy_error
import datetime
import matplotlib.pyplot as plt


class TwoLayerNet:
    def __init__(self, weight_init_std, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = x.dot(W1) + b1
        z1 = sigmoid(a1)
        a2 = z1.dot(W2) + b2
        out = softmax(a2)
        return out

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entrpy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y_labels = y.argmax(axis=1)
        return np.mean(y_labels == t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grad = {}
        grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grad


if __name__ == '__main__':
    N = 100
    D = 2
    X1 = np.random.randn(N, D) + np.array([4, 4])
    X2 = np.random.randn(N, D) + np.array([-4, 4])
    X3 = np.random.randn(N, D) + np.array([4, -4])
    X = np.concatenate([X1, X2, X3])
    t1 = np.zeros(N)
    t2 = np.zeros(N) + 1
    t3 = np.zeros(N) + 2
    t = np.concatenate([t1, t2, t3])
    t = np.array(t, dtype=int)

    X_indices = np.arange(3 * N)
    np.random.shuffle(X_indices)
    train_indices = X_indices[:210]
    test_indices = X_indices[210:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    t_train = t[train_indices]
    t_test = t[test_indices]

    network = TwoLayerNet(0.01, 2, 50, 3)
    prediction = network.predict(X)
    loss = network.loss(X, t)
    accuracy = network.accuracy(X, t)
    print(f"predicton  :{prediction}, loss : {loss}, accuracy : {accuracy}")

    loss_list = []
    accuracy_list = []

    batch_size = 10
    epochs = 3 * N / batch_size
    lr = 0.1
    for it in range(100):
        print("-" * 5 + f"{it} iteration" + "-" * 5)
        batch_mask = np.random.choice(X_train.shape[0], batch_size)
        x_batch = X_train[batch_mask]
        t_batch = t_train[batch_mask]

        start = datetime.datetime.now()
        grad = network.numerical_gradient(x_batch, t_batch)
        end = datetime.datetime.now()
        print(f"gradient calculation took {(end - start).seconds}")
        for key in network.params.keys():
            network.params[key] -= lr * grad[key]
        loss = network.loss(x_batch, t_batch)
        loss_list.append(loss)
        accuracy = network.accuracy(x_batch, t_batch)
        accuracy_list.append(accuracy)
        print(f"accuracy : {accuracy}, loss : {loss}")

    plt.plot(loss_list)
    plt.title("Loss")
    plt.figure()
    plt.plot(accuracy_list)
    plt.title("Accuracy")
    plt.show()
