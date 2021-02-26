import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    mask = (x <= 0)
    out = x.copy()
    out[mask] = 0
    return out


hidden_layer_size = 5
node_size = 100
N = 1000

x = np.random.randn(N, node_size)
x_initial = x
activations = {}
weights = {}

for i in range(hidden_layer_size):
    if i > 0:
        x = activations[i - 1]

    w = np.random.randn(node_size, node_size) * np.sqrt(2/node_size)
    weights[i] = w
    z = x.dot(w)
    activations[i] = relu(z)

for i, a in activations.items():
    plt.subplot(1, hidden_layer_size, i + 1)
    plt.title(f"{i + 1} layer")
    if i != 0:
        plt.yticks([], [])
    plt.hist(a.flatten(), 70)
    plt.ylim(0,30000)
    plt.xlim(0,3)
plt.show()
