import numpy as np
from practice.two_layer_net import TwoLayerNet
from practice.load_datasets import load_mnist
import matplotlib.pyplot as plt
import datetime

X_train, y_train, X_test, y_test = load_mnist()

network = TwoLayerNet(0.1, 784, 50, 10)
batch_size = 100
iter_num = 10
lr = 0.1

loss_list = []
accuracy_list = []

for it in range(iter_num):
    print("-" * 5 + f"{it+1}th iteration" + "-" * 5)
    batch_mask = np.random.choice(X_train.shape[0], batch_size)
    x_batch = X_train[batch_mask]
    t_batch = y_train[batch_mask]

    start = datetime.datetime.now()
    grad = network.numerical_gradient(x_batch, t_batch)
    end = datetime.datetime.now()
    print(f"gradient calculation took {(end - start).seconds} seconds")
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
