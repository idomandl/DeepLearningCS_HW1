import numpy as np
from metric_function import MetricFunction


class LinearLeastSquaresLoss(MetricFunction):
    name = "Linear Least Squares Loss"

    def __init__(self, Theta=None, Y=None):
        super().__init__(Theta, Y)

    def __call__(self, X, Y=None, Theta=None):
        MetricFunction.__call__(self, X, Y, Theta)
        # return 0.5 * sum(np.linalg.norm(x @ Theta - y)**2 for x, y in zip(X, Y)) / X.shape[0]
        # 0.5 * |Wx-b|^2
        return 0.5 * np.linalg.norm(X @ self.Theta - self.Y) ** 2 / X.shape[0]  # ? is it ok to divide by X.shape[0]?

    def calc_grad(self, X, Y=None, Theta=None):
        MetricFunction.calc_grad(self, X, Y, Theta)
        # return sum(x.T @ (x @ Theta - y) for x, y in zip(X, Y)) / X.shape[0]
        # (n, batch_size) @ (batch_size, m) = (n, m)
        return (X.T @ (X @ self.Theta - self.Y)) / X.shape[0]  # ? is it ok to divide by X.shape[0]?
