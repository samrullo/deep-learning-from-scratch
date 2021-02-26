import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from practice.multi_layer_with_batch_norm_option import MultiLayerNet
from practice.optimizers import RMSProp, SGD

(X_all_train, y_all_train), (X_test, y_test) = load_mnist(one_hot_label=True)

node_size = 100
layer_size = 5
output_size = 10

lr = 0.1
rho = 0.9
optimizer = SGD(lr)


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


def plot_accuracies(batch_norm_accuracies, non_batch_norm_accuracies, title):
    plt.figure()
    plt.plot(batch_norm_accuracies, label=f"batch norm accuracy")
    plt.plot(non_batch_norm_accuracies, linestyle="dashed", label=f"non-batch norm accuracy")
    plt.legend()
    plt.title(title)
    plt.show()


weight_init_std_list = [0.0001, 0.00018, 0.00034, 0.00063, 0.0011, 0.002, 0.003, 0.007, 0.01, 0.02, 0.04, 0.08, 0.158,
                        0.292, 1]

for weight_init_std in weight_init_std_list:
    print('-' * 5, weight_init_std, '-' * 5)
    batch_norm_nn = MultiLayerNet(weight_init_std, 784, 100, 10, 5, use_batch_norm=True)
    non_batch_norm_nn = MultiLayerNet(weight_init_std, 784, 100, 10, 5, use_batch_norm=False)
    batch_norm_losses, batch_norm_train_accuracies, batch_norm_test_accuracies = train_nn(batch_norm_nn)
    losses, _train_accuracies, _test_accuracies = train_nn(non_batch_norm_nn)
    plot_accuracies(batch_norm_test_accuracies, _test_accuracies, f"weight_init_std : {weight_init_std}")
