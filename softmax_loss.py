import numpy as np
from loss_function import LossFunction

class SoftmaxLossLoop(LossFunction):
    name = "SoftmaxLossLoop"

    def __init__(self, Theta, Y):
        super().__init__(Theta, Y)

    def __call__(self, X):
        sum_x = []
        for x in X:
            sum_k = []
            si = np.sum(np.exp(np.dot(x, self.Theta.T[i])) for i in range(self.Theta.T.shape[0]))
            for k, c in enumerate(self.Y.T):
                sum_k.append(np.log(np.exp(np.dot(x, self.Theta.T[k]))/si))
            sum_x.append(sum_k)
        return -np.sum(sum_x)/X.shape[0]

    def calc_grad(self, X):
        # sum_x = []
        # for x in X:
        #     sum_k = []
        #     for k,c in enumerate(self.Y.T):
        #         sum_k.append(np.exp(np.dot(x, self.Theta.T[k]))/np.sum(np.exp(np.dot(x, self.Theta.T[i])) for i in range(self.Theta.T.shape[0])) - c)
        #     sum_x.append(sum_k)
        # return X.T @ np.sum(sum_x) / X.shape[0]
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T) - self.Y) / X.shape[0]
class SoftmaxLoss(LossFunction):

    name = ("SoftmaxLos"
            "s")
    def __init__(self, Theta, Y):
        super().__init__(Theta, Y)

    def __call__(self, X):
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # sum((batch_size, m) * (batch_size, m)) / batch_size
        return np.sum(self.Y * np.log(np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))) / X.shape[0]

    def calc_grad(self, X):
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T) - self.Y) / X.shape[0]
