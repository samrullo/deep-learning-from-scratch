import numpy as np


class BatchNormalization:
    def __init__(self, gamma, beta, rho=0.9, moving_mean=None, moving_var=None):
        self.gamma = gamma
        self.beta = beta
        self.rho = rho
        self.moving_mean = moving_mean
        self.moving_var = moving_var

        self.batch_size = None
        self.x_mu = None
        self.x_std = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, training_flg=True, epsilon=1e-8):
        if (self.moving_mean is None) or (self.moving_var is None):
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)

        if training_flg:
            mu = np.mean(x, axis=0)
            x_mu = x - mu
            var = np.mean(x_mu ** 2, axis=0)
            std = np.sqrt(var + epsilon)
            x_std = x_mu / std
            self.moving_mean = self.rho * mu + (1 - self.rho) * mu
            self.moving_var = self.rho * var + (1 - self.rho) * var

            # set instance variables
            self.x_mu = x_mu
            self.std = std
        else:
            x_mu = x - self.moving_mean
            x_std = x_mu / np.sqrt(self.moving_var + epsilon)
        out = self.gamma * x_std + self.beta
        return out

    def backward(self, dout):
        N, D = self.x_mu.shape

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(self.x_mu * dout, axis=0)

        a1 = self.gamma * dout
        a2 = a1 / self.std
        a3 = self.x_mu * a1
        a3 = np.sum(a3, axis=0)
        a4 = -a3 / (self.std * self.std)
        a5 = 0.5 * a4 / self.std
        a6 = a5 / N
        a7 = 2 * self.x_mu * a6
        a8 = -(a2 + a7)
        a9 = a8 / N
        dx = a2 + a7 + a9

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx
