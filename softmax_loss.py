import numpy as np
from loss_function import LossFunction

class SoftmaxLossLoop(LossFunction):
    name = "SoftmaxLossLoop"

    def __init__(self, Theta, Y):
        super().__init__(Theta, Y)

    def __call__(self, X):
        # sum_x = []
        # for x in X:
        #     sum_k = []
        #     si = np.sum(np.exp(np.dot(x, self.Theta.T[i])) for i in range(self.Theta.T.shape[0]))
        #     for k, c in enumerate(self.Y.T):
        #         sum_k.append(np.log(np.exp(np.dot(x, self.Theta.T[k]))/si))
        #     sum_x.append(sum_k)
        #print('$$$$$$$$$$$$$$$$$')
        #print(-np.sum(sum_x)/X.shape[0])
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # sum((batch_size, m) * (batch_size, m)) / batch_size
        #print(np.sum(self.Y * np.log(np.exp(X @ self.Theta - max_element) / sum(
        #    np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))) / X.shape[0])
        #print('$$$$$$$$$$$$$$$$$')
        return np.sum(self.Y * np.log(np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))) / X.shape[0]
        return -np.sum(sum_x)/X.shape[0]

    def calc_grad(self, X):
        grad = [0]*self.Theta.T.shape[0]
        for p, w_p in enumerate(self.Theta.T):
            grad[p] = (-1/X.shape[0])* X.T @ (self.Y.T[p].T - np.sum([np.linalg.inv(np.diag(np.sum([np.exp(X @ w_j.T) for w_j in self.Theta.T], axis=0))) @ np.exp(X @ w_p.T) * c_k.T for c_k in self.Y.T], axis=0))
        print('---------------')
        print(np.array(grad).T)
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        print(X.T @ (np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T) - self.Y) / X.shape[0])
        print(self.Y)
        print('---------------')
        #return np.array(grad).T
        # max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # # (n, batch_size) @ (batch_size, m) = (n, m)
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
        return -(np.sum(self.Y * np.log(np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T))) / X.shape[0])

    def calc_grad(self, X):
        max_element = (X @ self.Theta).max(axis=1, keepdims=True)
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (np.exp(X @ self.Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in self.Theta.T) - self.Y) / X.shape[0]

