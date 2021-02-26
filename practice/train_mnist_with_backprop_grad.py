from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
from practice.two_layer_net_backprop import TwoLayerNetBackprop
from practice.optimizers import NesterovAG,SGD

(X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

print(f"X_train shape : {X_train.shape}, y_train shape : {y_train.shape}")

network = TwoLayerNetBackprop(0.01, 784, 50, 10)
x_batch = X_train[:3]
y_batch = y_train[:3]

num_grads = network.numerical_gradient(x_batch, y_batch)
backprop_grads = network.gradient(x_batch, y_batch)

for key in num_grads.keys():
    diff = np.mean(np.abs(num_grads[key] - backprop_grads[key]))
    print(f"{key} diff : {diff}")

batch_size = 100
train_size = X_train.shape[0]
epochs = int(X_train.shape[0] / batch_size)
lr = 0.1
momentum = 0.8
loss_list = []
train_accuracy_list = []
test_accuracy_list = []

num_iters = 10000
iters_per_epoch = max(train_size / batch_size, 1)

optimizer = SGD(lr)
for it in range(num_iters):

    batch_mask = np.random.choice(X_train.shape[0], batch_size)
    x_batch = X_train[batch_mask]
    t_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)

    optimizer.update(network.params, grads)
    loss = network.loss(x_batch, t_batch)
    loss_list.append(loss)
    train_accuracy = network.accuracy(x_batch, t_batch)
    train_accuracy_list.append(train_accuracy)
    test_accuracy = network.accuracy(X_test, y_test)
    test_accuracy_list.append(test_accuracy)
    if it % iters_per_epoch == 0:
        print("-" * 5 + f"{round(it / iters_per_epoch)}th epoch" + "-" * 5)
        print(f"train accuracy : {train_accuracy}, test accuracy : {test_accuracy} loss : {loss}")

plt.plot(loss_list)
plt.title("Loss")
plt.figure()
plt.plot(train_accuracy_list, label='train accuracy')
plt.plot(test_accuracy_list, linestyle='dashed', label="test accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()
