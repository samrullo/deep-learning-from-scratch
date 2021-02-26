import numpy as np


def train_nn(network, X_all_train, y_all_train, X_test, y_test, optimizer, epochs=10):
    """
    Train neural network
    :param network:
    :param X_all_train: train data
    :param y_all_train: train labels
    :param X_test: test data
    :param y_test: test labels
    :param optimizer: optimizer
    :param epochs: number of epochs
    :return: losses,train_accuracies,test_accuracies
    """
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
