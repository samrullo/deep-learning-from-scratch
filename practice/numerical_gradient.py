import numpy as np


def numerical_grad_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # gradient will have the same shape as x

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_grad_no_batch(f, x)
    grad = np.zeros_like(x)
    for idx, x_row in enumerate(x):
        grad[idx] = numerical_grad_no_batch(f, x_row)
    return grad


if __name__ == '__main__':
    def simple_func(x):
        if x.ndim == 1:
            return (x ** 2).sum()
        return (x ** 2).sum(axis=1)


    print(f"let's calculate gradient at [-4,4]")
    grad = numerical_grad_no_batch(simple_func, np.array([-4, 4], dtype=float))
    print(f"grad is {grad}")

    x = 4 * np.random.randn(11, 2)
    grad_batch = numerical_gradient(simple_func, x)
    for idx, grad_row in enumerate(grad_batch):
        print(f"gradient of {x[idx]} is {grad_row}")
