import numpy as np


def cross_entrpy_error(y, t):
    if t.ndim == 2:
        t = t.argmax(axis=1)
    batch_size = t.shape[0]
    h = 1e-4
    return -np.sum(np.log(y[np.arange(batch_size), t] + h)) / batch_size


if __name__ == '__main__':
    from practice.activation import softmax
    import matplotlib.pyplot as plt

    accuracy_list = []
    loss_list = []

    for it in range(10):
        x = np.random.randn(11, 3)
        y = softmax(x)
        t = np.random.randint(0, 2, x.shape[0])
        loss = cross_entrpy_error(y, t)
        y_labels = y.argmax(axis=1)
        print("-" * 5 + f"{it + 1}" + "-" * 5)
        print(f"y is {y_labels}")
        print(f"t is {t}")
        accuracy = np.mean(y_labels == t)
        print(f"accuracy is {accuracy}")
        print(f"loss is {loss}")
        print("-" * 11)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
    plt.scatter(accuracy_list, loss_list)
    plt.xlabel('accuracy')
    plt.ylabel('loss')
    plt.title('Accuracy vs Loss')
    plt.show()
