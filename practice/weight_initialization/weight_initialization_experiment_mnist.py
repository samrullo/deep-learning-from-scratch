import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from practice.weight_initialization.multi_layer_net import MultiLayerNet
from practice.optimizers import RMSProp, SGD

(X_all_train, y_all_train), (X_test, y_test) = load_mnist(one_hot_label=True)

node_size = 100
layer_size = 5
output_size = 10

lr = 0.1
rho = 0.9
optimizer = SGD(lr)


def train_nn(network):
    epochs = 10
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


def plot_losses_and_accuracies(losses, accuracies, label):
    plt.figure()
    plt.plot(losses, label=f"{label} losses")
    plt.plot(accuracies, linestyle="dashed", label=f"{label} accuracies")
    plt.legend()
    plt.show()


def plot_train_test_accuracies(train_accuracies, test_accuracies, label):
    plt.figure()
    plt.plot(train_accuracies, label=f"{label} train accuracy")
    plt.plot(test_accuracies, linestyle="dashed", label=f"{label} test accuracy")
    plt.legend()
    plt.show()


def plot_weight_hist(network, title):
    fig, axs = plt.subplots(1, layer_size + 1)
    fig.suptitle(title)
    for i in range(layer_size + 1):
        axs[i].hist(network.params[f"W{i + 1}"].flatten(), 30, range=(0, 1))
        if i != 0:
            axs[i].set_yticks([])
            axs[i].set_yticklabels([])
        axs[i].set_title(f"{i + 1} layer")
    plt.show()


networks = {'naive_wi': MultiLayerNet('naive_wi', X_all_train.shape[1], node_size, output_size, layer_size),
            'xavier': MultiLayerNet('xavier', X_all_train.shape[1], node_size, output_size, layer_size),
            'he': MultiLayerNet('he', X_all_train.shape[1], node_size, output_size, layer_size)}

train_accuracies_dict = {}
test_accuracies_dict = {}

for key, network in networks.items():
    losses_, train_accuracies, test_accuracies = train_nn(network)
    train_accuracies_dict[key] = train_accuracies
    test_accuracies_dict[key] = test_accuracies
    # plot_train_test_accuracies(train_accuracies, test_accuracies, key)
    # plot_losses_and_accuracies(losses_, accuracies_, f'{key} weight init')
    # plot_weight_hist(network, f'{key} weight initialization')

for key in train_accuracies_dict.keys():
    plot_train_test_accuracies(train_accuracies_dict[key], test_accuracies_dict[key], key)
