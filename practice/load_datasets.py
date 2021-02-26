import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from skimage import io
import matplotlib.pyplot as plt


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"X_train shape : {X_train.shape}, y_train shape : {y_train.shape}")
    from practice.two_layer_net import TwoLayerNet

    network = TwoLayerNet(0.1, 784, 50, 10)
    x_traind_idx = 1
    prediction = network.predict(X_train[x_traind_idx])
    print(f"prediction of below image is {prediction.argmax()}")
    io.imshow(X_train[x_traind_idx].reshape(28, 28))
    plt.show()
    grads=network.numerical_gradient(X_train[:3],y_train[:3])
