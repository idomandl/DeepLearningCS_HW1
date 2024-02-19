import numpy as np
from function.function import Function


class Softmax(Function):
    name = "Softmax"

    def __call__(self, X, Theta):
        # return np.exp(X @ self.Theta) / np.sum(np.exp(X @ self.Theta), axis=1, keepdims=True)
        max_element = (X @ Theta).max(axis=1, keepdims=True)
        return np.log(np.exp(X @ Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in Theta.T))

    def loss(self, X, Y, Theta):
        max_element = (X @ Theta).max(axis=1, keepdims=True)
        # sum((batch_size, m) * (batch_size, m)) / batch_size
        return -(np.sum(Y * np.log(np.exp(X @ Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in Theta.T))) / X.shape[0])

    def loss_grad_Theta(self, X, Y, Theta):
        max_element = (X @ Theta).max(axis=1, keepdims=True)
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return X.T @ (np.exp(X @ Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in Theta.T) - Y) / X.shape[0]

    def loss_grad_X(self, X, Y, Theta):
        max_element = (X @ Theta).max(axis=1, keepdims=True)
        # ((n, m) @ (batch_size, m).T).T = (n, batch_size).T = (batch_size, n)
        return (Theta @ (np.exp(X @ Theta - max_element) / sum(
            np.exp(X @ Theta_i.T[:, None] - max_element) for Theta_i in Theta.T) - Y).T).T / X.shape[0]
