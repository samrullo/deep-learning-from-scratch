import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from practice.multi_layer_with_weight_decay import MultiLayerNetL2Norm
from practice.multi_layer_with_batch_norm_option import MultiLayerNet
from practice.optimizers import RMSProp, SGD

(X_all_train, y_all_train), (X_test, y_test) = load_mnist(one_hot_label=True)

node_size = 100
layer_size = 5
output_size = 10

lr = 0.1
rho = 0.9
optimizer = SGD(lr)


def train_nn_on_whole_dataset(network, x_train, y_train, epochs=11):
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        grads = network.gradient(x_train, y_train)
        optimizer.update(network.params, grads)
        train_accuracy = network.accuracy(x_train, y_train)
        test_accuracy = network.accuracy(X_test, y_test)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            f"epoch {epoch + 1} loss : {network.loss(x_train, y_train)}, train accuracy : {train_accuracy}, test accuracy : {test_accuracy} ")
    return train_accuracies, test_accuracies


def train_nn(network):
    epochs = 2
    train_size = X_all_train.shape[0]
    batch_size = 100
    minibatch_num = np.ceil(train_size / batch_size).astype(int)
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        idx = np.arange(train_size)
        np.random.shuffle(idx)
        for mn in range(minibatch_num):
            batch_mask = idx[batch_size * mn:batch_size * (mn + 1)]
            x_batch = X_all_train[batch_mask]
            y_batch = y_all_train[batch_mask]
            grads = network.gradient(x_batch, y_batch)
            optimizer.update(network.params, grads)
            if mn % 100 == 0:
                train_accuracies.append(network.accuracy(X_all_train, y_all_train))
                test_accuracies.append(network.accuracy(X_test, y_test))
        print(
            f"epoch {epoch + 1} loss : {network.loss(x_batch, y_batch)}, accuracy : {network.accuracy(x_batch, y_batch)} ")
        losses.append(network.loss(x_batch, y_batch))
    return losses, train_accuracies, test_accuracies


def plot_train_test_accuracies(train_accuracies, test_accuracies, label):
    plt.figure()
    plt.plot(train_accuracies, label=f"{label} train accuracy")
    plt.plot(test_accuracies, linestyle="dashed", label=f"{label} test accuracy")
    plt.legend()
    plt.show()


network_with_l2_norm = MultiLayerNetL2Norm(weight_init_std=np.sqrt(2 / node_size), input_size=784, node_size=100,
                                           output_size=10,
                                           layer_size=6, l2_norm_lambda=0.01, use_batch_norm=False)
network = MultiLayerNet(weight_init_std=np.sqrt(2 / node_size), input_size=784, node_size=100, output_size=10,
                        layer_size=6,
                        use_batch_norm=False)

np.random.seed(1111)
idx = np.arange(X_all_train.shape[0])
np.random.shuffle(idx)
train_idx = idx[:300]
x_train = X_all_train[train_idx]
y_train = y_all_train[train_idx]

train_accuracies, test_accuracies = train_nn_on_whole_dataset(network, x_train, y_train, epochs=200)
l2_train_accuracies, l2_test_accuracies = train_nn_on_whole_dataset(network_with_l2_norm, x_train, y_train, epochs=200)

plot_train_test_accuracies(train_accuracies, test_accuracies, 'Without L2 Normalization')
plot_train_test_accuracies(l2_train_accuracies, l2_test_accuracies, 'L2 Normalization')
