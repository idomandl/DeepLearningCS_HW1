import numpy as np
from function.function import Function


class LinearLeastSquares(Function):
    name = "Linear Least Squares"

    def loss(self, X, Y, Theta):
        # return 0.5 * sum(np.linalg.norm(x @ Theta - y)**2 for x, y in zip(X, Y)) / X.shape[0]
        # 0.5 * |Wx-b|^2
        return 0.5 * np.linalg.norm(X @ Theta - Y) ** 2 / X.shape[0]

    def loss_grad_Theta(self, X, Y, Theta):
        # return sum(x.T @ (x @ Theta - y) for x, y in zip(X, Y)) / X.shape[0]
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return (X.T @ (X @ Theta - Y)) / X.shape[0]
