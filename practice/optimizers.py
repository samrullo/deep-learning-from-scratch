import matplotlib.pyplot as plt
import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, momentum, lr):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if not self.v:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class NesterovAG:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            v_pre = self.v[key]
            self.v[key] = self.momentum * v_pre - self.lr * grads[key]
            params[key] = params[key] - self.momentum * v_pre + (1 + self.momentum) * self.v[key]


class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None
        self.epsilon = 1e-7

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = self.h[key] + grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (self.epsilon + np.sqrt(self.h[key]))


class RMSProp:
    def __init__(self, lr, rho):
        self.lr = lr
        self.rho = rho
        self.h = None
        self.epsilon = 1e-7

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]
            params[key] += -self.lr * grads[key] / (np.sqrt(self.epsilon + self.h[key]))


class AdaDelta:
    def __init__(self, rho):
        self.rho = rho
        self.epsilon = 1e-4
        self.h = None
        self.r = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            self.r = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                self.r[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]
            rms_param = np.sqrt(self.r[key] + self.epsilon)
            rms_grads = np.sqrt(self.h[key] + self.epsilon)
            dp = -(rms_param / rms_grads) * grads[key]
            params[key] += dp
            self.r[key] = self.rho * self.r[key] + (1 - self.rho) * dp * dp


class Adam:
    def __init__(self, lr, rho1, rho2):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.epsilon = 1e-7
        self.m = None
        self.h = None
        self.iter = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.h = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.iter += 1
        for key in params.keys():
            self.m[key] = self.rho1 * self.m[key] + (1 - self.rho1) * grads[key]
            self.h[key] = self.rho2 * self.h[key] + (1 - self.rho2) * grads[key] * grads[key]
            m_hat = self.m[key] / (1 + np.power(self.rho1, self.iter))
            h_hat = self.h[key] / (1 + np.power(self.rho2, self.iter))
            params[key] -= self.lr * m_hat / (self.epsilon + np.sqrt(h_hat))


if __name__ == '__main__':
    from practice.simple_layers import SimpleSquareFuncLayer

    # start at a point -3,3
    params = {'x1': -3, 'x2': -2}
    lr = 0.1
    sgd = SGD(lr)

    # do 17 iterations
    loss_list = []
    x1_list = []
    x2_list = []
    iters_num = 17
    for it in range(iters_num):
        simpleLayer = SimpleSquareFuncLayer(params['x1'], params['x2'])
        x1_list.append(params['x1'])
        x2_list.append(params['x2'])
        loss_list.append(simpleLayer.forward())
        dx1, dx2 = simpleLayer.backward(1)
        grads = {'x1': dx1, 'x2': dx2}
        sgd.update(params, grads)
    print(f"final paarmeters : {params}")
    x1_arr = np.array(x1_list)
    x2_arr = np.array(x2_list)
    x1_coords = np.arange(-3, 3, 0.1)
    x2_coords = np.arange(-3, 3, 0.1)
    mx1, mx2 = np.meshgrid(x1_coords, x2_coords)
    simpleLayer = SimpleSquareFuncLayer(mx1, mx2)
    mout = simpleLayer.forward()
    plt.pcolormesh(mx1, mx2, mout, cmap='jet')
    plt.colorbar()
    plt.scatter(x1_arr, x2_arr, c='w')
    plt.show()
